from typing import Any, ClassVar

import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

"""
The bulk of this file is copied from https://github.com/ApolloResearch/e2e_sae
licensed under MIT, (c) 2024 ApolloResearch.
"""


class DatasetConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    name: str = "lennart-finke/SimpleStories"
    is_tokenized: bool = True
    hf_tokenizer_path: str | None = None
    streaming: bool = False
    split: str = "train"
    n_ctx: int = 1024
    seed: int | None = None
    column_name: str = "input_ids"
    """The name of the column in the dataset that contains the data (tokenized or non-tokenized).
    Typically 'input_ids' for datasets stored with e2e_sae/scripts/upload_hf_dataset.py, or "tokens"
    for datasets tokenized in TransformerLens (e.g. NeelNanda/pile-10k)."""


def _keep_single_column(dataset: Dataset, col_name: str) -> Dataset:
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful
    when we want to tokenize and mix together different strings.
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    column_name: str,
    max_length: int = 1024,
    add_bos_token: bool = False,
    num_proc: int = 10,
    to_lower: bool = False,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to
    tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of
    shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if
    parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with
    padding, then remove padding at the end.

    NOTE: Adapted from
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utils.py#L267
    to handle IterableDataset.

    TODO: Fix typing of tokenizer

    This tokenization is useful for training language models, as it allows us to efficiently train
    on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding).
    Further, for models with absolute positional encodings, this avoids privileging early tokens
    (eg, news articles often begin with CNN, and models may learn to use early positional
    encodings to predict these)

    Args:
        dataset: The dataset to tokenize, assumed to be a HuggingFace text dataset. Can be a regular
            Dataset or an IterableDataset.
        tokenizer: The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        max_length: The length of the context window of the sequence. Defaults to 1024.
        column_name: The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token: Add BOS token at the beginning of each sequence. Defaults to False as this
            is not done during training.

    Returns:
        Dataset or IterableDataset: Returns the tokenized dataset, as a dataset of tensors, with a
        single column called "input_ids".

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it
    just outputs nothing. I'm not super sure why
    """
    dataset = _keep_single_column(dataset, column_name)
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(
        examples: dict[str, list[str]],
    ) -> dict[
        str,
        NDArray[np.signedinteger[Any]],
    ]:
        text = examples[column_name]
        # Concatenate all the text into a single string, separated by EOS tokens
        assert hasattr(tokenizer, "eos_token") and isinstance(tokenizer.eos_token, str)
        full_text = tokenizer.eos_token.join(text)

        # Split the text into chunks for parallel tokenization
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]

        # Tokenize the chunks using the Tokenizer library
        if to_lower:
            chunks = [
                chunk.replace(tokenizer.eos_token.lower(), tokenizer.eos_token) for chunk in chunks
            ]
        tokens = [tokenizer.encode(chunk) for chunk in chunks]  # Get token IDs for each chunk
        tokens = np.concatenate(tokens)  # Flatten the list of token IDs

        # Calculate number of batches and adjust the tokens accordingly
        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]

        # Reshape tokens into batches
        tokens = tokens.reshape((num_batches, seq_len))

        # Optionally, add BOS token at the beginning of each sequence
        if add_bos_token:
            assert hasattr(tokenizer, "bos_token_id")
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)

        return {"input_ids": tokens}

    # Apply the tokenization function to the dataset
    if isinstance(dataset, IterableDataset):
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name]
        )
    else:
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name], num_proc=num_proc
        )

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def create_data_loader(
    dataset_config: DatasetConfig,
    batch_size: int,
    buffer_size: int = 1000,
    global_seed: int = 0,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    to_lower: bool = True,
) -> tuple[DataLoader[Any], PreTrainedTokenizer]:
    """Create a DataLoader for the given dataset.

    Args:
        dataset_config: The configuration for the dataset.
        batch_size: The batch size.
        buffer_size: The buffer size for streaming datasets.
        global_seed: Used for shuffling if dataset_config.seed is None.
        ddp_rank: The rank of the current process in DDP.
        ddp_world_size: The world size in DDP.

    Returns:
        A tuple of the DataLoader and the tokenizer.
    """
    dataset = load_dataset(
        dataset_config.name,
        streaming=dataset_config.streaming,
        split=dataset_config.split,
        trust_remote_code=False,
    )
    seed = dataset_config.seed if dataset_config.seed is not None else global_seed
    if dataset_config.streaming:
        assert isinstance(dataset, IterableDataset)
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    else:
        assert isinstance(dataset, Dataset)
        dataset = dataset.shuffle(seed=seed)
    dataset = split_dataset_by_node(dataset, ddp_rank, ddp_world_size)  # pyright: ignore[reportArgumentType]

    tokenizer = AutoTokenizer.from_pretrained(dataset_config.hf_tokenizer_path)

    torch_dataset: Dataset | IterableDataset
    if dataset_config.is_tokenized:
        torch_dataset = dataset.with_format("torch")
        # Get a sample from the dataset and check if it's tokenized and what the n_ctx is
        # Note that the dataset may be streamed, so we can't just index into it
        sample = next(iter(torch_dataset))[dataset_config.column_name]
        assert isinstance(sample, torch.Tensor) and sample.ndim == 1, (
            "Expected the dataset to be tokenized."
        )
        assert len(sample) == dataset_config.n_ctx, "n_ctx does not match the tokenized length."

    else:
        to_lower = "SimpleStories" in dataset_config.name
        torch_dataset = tokenize_and_concatenate(
            dataset,
            tokenizer,
            max_length=dataset_config.n_ctx,
            column_name=dataset_config.column_name,
            add_bos_token=False,
            to_lower=to_lower,
        )

    loader = DataLoader[Dataset | IterableDataset](
        torch_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return loader, tokenizer
