from collections.abc import Iterator
from typing import Generic, Literal, TypeVar, cast, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

Q = TypeVar("Q")


class DatasetGeneratedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that generates batches by calling the dataset's `generate_batch` method."""

    def __init__(
        self,
        dataset: Dataset[Q],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        # assert that dataset has a generate_batch method
        assert hasattr(dataset, "generate_batch")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @override
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[Q]:
        for _ in range(len(self)):
            yield self.dataset.generate_batch(self.batch_size)  # pyright: ignore[reportAttributeAccessIssue]


class BatchedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that unpacks the batch in __getitem__.

    This is used for datasets which generate a whole batch in one call to __getitem__.
    """

    def __init__(
        self,
        dataset: Dataset[Q],
        num_workers: int = 0,
    ):
        super().__init__(dataset, num_workers=num_workers)

    @override
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        for batch, label in super().__iter__():
            yield batch[0], label[0]


class InductionDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch seq_len"],
            Float[Tensor, "batch 1"],
        ]
    ]
):
    """
    Generates data of the format TTTTTSMTTT...SM
    where T is a token from the base vocabulary, S is a special induction token,
    and M is a memorised token that appears twice in the sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        device: str | torch.device,
        prefix_window: int,
        size: int = 100_000,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.prefix_window = prefix_window
        self.size = size
        self.induction_token = vocab_size + 1  # One additional token for the induction token
        self.device = device
        assert self.prefix_window < seq_len - 2, "S M … S M must fit."

    def __len__(self) -> int:
        return 2**31

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_start = 1  # 0 is reserved for BOS
        vocab_end = self.vocab_size + vocab_start

        memorised_token = torch.randint(vocab_start, vocab_end, (batch_size, 1), dtype=torch.long)
        tokens = torch.randint(
            vocab_start, vocab_end, (batch_size, self.seq_len - 2), dtype=torch.long
        )
        memory_positions_first = torch.randint(
            1, self.prefix_window, (batch_size, 1), dtype=torch.long
        )

        tokens.scatter_(1, memory_positions_first, self.induction_token)
        tokens.scatter_(1, memory_positions_first + 1, memorised_token)

        tokens = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.long),  # BOS token
                tokens,
                self.induction_token * torch.ones((batch_size, 1), dtype=torch.long),
            ),
            dim=1,
        )

        # Label is the memorised token that appears twice
        return tokens.to(self.device), memorised_token.to(self.device).squeeze(-1)


class TrigramDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch seq_len"],
            Float[Tensor, "batch 1"],
        ]
    ]
):
    """
    Generates data with skip-trigram patterns like ATTTTBCTTTXYZCTTT
    where T is a random token, A...BC and X...YZ are trigram relationships
    with BC and YZ being adjacent tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        device: str | torch.device,
        n_trigrams: int = 10,
        min_skip_distance: int = 3,
        max_skip_distance: int = 10,
        size: int = 100_000,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_trigrams = n_trigrams
        self.min_skip_distance = min_skip_distance
        self.max_skip_distance = max_skip_distance
        self.size = size
        self.device = device

        vocab_start = 1  # 0 is reserved for BOS
        vocab_end = self.vocab_size + vocab_start
        # creates 3 x n_trigrams matrix

        # Pre-generate n_trigrams trigram relationships: (A, B) -> C
        self.trigram_first = torch.randint(vocab_start, vocab_end, (n_trigrams,), dtype=torch.long)
        self.trigram_second = torch.randint(vocab_start, vocab_end, (n_trigrams,), dtype=torch.long)
        self.trigram_third = torch.randint(vocab_start, vocab_end, (n_trigrams,), dtype=torch.long)

        # Ensure trigram tokens are distinct within each trigram
        for i in range(n_trigrams):
            while self.trigram_second[i] == self.trigram_first[i]:
                self.trigram_second[i] = torch.randint(
                    vocab_start, vocab_end, (1,), dtype=torch.long
                )
            while self.trigram_third[i] in [self.trigram_first[i], self.trigram_second[i]]:
                self.trigram_third[i] = torch.randint(
                    vocab_start, vocab_end, (1,), dtype=torch.long
                )
        assert seq_len >= max_skip_distance + 4, (
            "Sequence must be long enough for skip-trigrams (BOS + A + skip + BC)"
        )

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_start = 1  # 0 is reserved for BOS
        vocab_end = self.vocab_size + vocab_start

        # Choose random trigrams for each sequence in the batch
        chosen_trigrams = torch.randint(0, self.n_trigrams, (batch_size,), dtype=torch.long)

        # Get the trigram tokens for each sequence
        first_tokens = self.trigram_first[chosen_trigrams]
        second_tokens = self.trigram_second[chosen_trigrams]
        third_tokens = self.trigram_third[chosen_trigrams]

        sequences = []

        for i in range(batch_size):
            # Generate skip distance between A and BC
            #
            skip_distance: int = cast(
                int, torch.randint(self.min_skip_distance, self.max_skip_distance + 1, (1,)).item()
            )

            # Calculate how many tokens we can put before A
            trigram_part_length = 1 + skip_distance + 2  # A + skip + BC
            remaining_length = self.seq_len - 1 - trigram_part_length  # -1 for BOS

            prefix_length: int = cast(int, torch.randint(0, (remaining_length + 1), (1,)).item())
            suffix_length = remaining_length - prefix_length

            # Create the sequence parts
            sequence_parts = []

            # Generate trigram tokens set for exclusion
            trigram_tokens = {
                first_tokens[i].item(),
                second_tokens[i].item(),
                third_tokens[i].item(),
            }

            # Prefix (random tokens before A)
            if prefix_length > 0:
                prefix = torch.randint(vocab_start, vocab_end, (prefix_length,), dtype=torch.long)
                # Ensure prefix tokens are different from trigram tokens
                for j in range(prefix_length):
                    while prefix[j].item() in trigram_tokens:
                        prefix[j] = torch.randint(vocab_start, vocab_end, (1,), dtype=torch.long)
                sequence_parts.append(prefix)

            # A token
            sequence_parts.append(first_tokens[i : i + 1])

            # Skip tokens between A and BC
            if skip_distance > 0:
                skip_tokens = torch.randint(
                    vocab_start, vocab_end, (skip_distance,), dtype=torch.long
                )
                # Ensure skip tokens are different from trigram tokens
                for j in range(skip_distance):
                    while skip_tokens[j].item() in trigram_tokens:
                        skip_tokens[j] = torch.randint(
                            vocab_start, vocab_end, (1,), dtype=torch.long
                        )
                sequence_parts.append(skip_tokens)

            # BC tokens (adjacent)
            sequence_parts.append(torch.stack([second_tokens[i], third_tokens[i]]))

            # Suffix (random tokens after BC)
            if suffix_length > 0:
                suffix = torch.randint(vocab_start, vocab_end, (suffix_length,), dtype=torch.long)
                # Ensure suffix tokens are different from trigram tokens
                for j in range(suffix_length):
                    while suffix[j].item() in trigram_tokens:
                        suffix[j] = torch.randint(vocab_start, vocab_end, (1,), dtype=torch.long)
                sequence_parts.append(suffix)

            # Concatenate all parts
            sequence = torch.cat(sequence_parts, dim=0)
            sequences.append(sequence)

        # Stack all sequences
        tokens = torch.stack(sequences, dim=0)

        # Add BOS token at the beginning
        tokens = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.long),  # BOS token
                tokens,
            ),
            dim=1,
        )

        # Label is the third token of the trigram (the predicted token)
        return tokens.to(self.device), third_tokens.to(self.device)


DataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]


class SparseFeatureDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch n_features"],
            Float[Tensor, "batch n_features"],
        ]
    ]
):
    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: list[list[int]] | None = None,
    ):
        self.n_features: int = n_features
        self.feature_probability: float = feature_probability
        self.device: str = device
        self.data_generation_type: DataGenerationType = data_generation_type
        self.value_range: tuple[float, float] = value_range
        self.synced_inputs: list[list[int]] | None = synced_inputs

    def __len__(self) -> int:
        return 2**31

    def sync_inputs(
        self, batch: Float[Tensor, "batch n_features"]
    ) -> Float[Tensor, "batch n_features"]:
        assert self.synced_inputs is not None
        all_indices = [item for sublist in self.synced_inputs for item in sublist]
        assert len(all_indices) == len(set(all_indices)), "Synced inputs must be non-overlapping"
        for indices in self.synced_inputs:
            mask = torch.zeros_like(batch, dtype=torch.bool)
            # First, get the samples for which there is a non-zero value for any of the indices
            non_zero_samples = (batch[..., indices] != 0.0).any(dim=-1)
            for idx in indices:
                mask[..., idx] = non_zero_samples
            # Now generate random values in value_range and apply them to the masked elements
            max_val, min_val = self.value_range
            random_values = torch.rand(batch.shape[0], self.n_features, device=self.device)
            random_values = random_values * (max_val - min_val) + min_val
            batch = torch.where(mask, random_values, batch)
        return batch

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
        # TODO: This is a hack to keep backward compatibility. Probably best to have
        # data_generation_type: Literal["exactly_n_active", "at_least_zero_active"] and
        # data_generation_n: PositiveInt
        number_map = {
            "exactly_one_active": 1,
            "exactly_two_active": 2,
            "exactly_three_active": 3,
            "exactly_four_active": 4,
            "exactly_five_active": 5,
        }
        if self.data_generation_type in number_map:
            n = number_map[self.data_generation_type]
            batch = self._generate_n_feature_active_batch(batch_size, n=n)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._masked_batch_generator(batch_size)
            if self.synced_inputs is not None:
                batch = self.sync_inputs(batch)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        return batch, batch.clone().detach()

    def _generate_n_feature_active_batch(
        self, batch_size: int, n: int
    ) -> Float[Tensor, "batch n_features"]:
        """Generate a batch with exactly n features active per sample.

        Args:
            batch_size: Number of samples in the batch
            n: Number of features to activate per sample
        """
        if n > self.n_features:
            raise ValueError(
                f"Cannot activate {n} features when only {self.n_features} features exist"
            )

        batch = torch.zeros(batch_size, self.n_features, device=self.device)

        # Create indices for all features
        feature_indices = torch.arange(self.n_features, device=self.device)
        # Expand to batch size
        feature_indices = feature_indices.expand(batch_size, self.n_features)

        # For each instance in the batch, randomly permute the features
        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)

        # Take first n indices for each instance - guaranteed no duplicates
        active_features = permuted_features[..., :n]

        # Generate random values in value_range for the active features
        min_val, max_val = self.value_range
        random_values = torch.rand(batch_size, n, device=self.device)
        random_values = random_values * (max_val - min_val) + min_val

        # Place each active feature
        for i in range(n):
            batch.scatter_(
                dim=1, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1]
            )

        return batch

    def _masked_batch_generator(self, batch_size: int) -> Float[Tensor, "batch_size n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.
        """
        min_val, max_val = self.value_range
        batch = (
            torch.rand((batch_size, self.n_features), device=self.device) * (max_val - min_val)
            + min_val
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask
