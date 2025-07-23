from collections.abc import Iterator
from typing import Generic, Literal, TypeVar, override

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
    Generates data with skip-trigram patterns like ATTTTBC
    where T is a random token, A...BC is a trigram relationship
    and BC are adjacent tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        device: str | torch.device,
        n_trigrams: int = 10,
        min_skip_distance: int = 1,
        max_skip_distance: int = 10,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_trigrams = n_trigrams
        self.min_skip_distance = min_skip_distance
        self.max_skip_distance = max_skip_distance
        self.device = device

        # Randomly choose trigrams from the vocabulary. Each trigram is unique for simplicity's sake
        trigram_choices = torch.randperm(vocab_size)[1 : n_trigrams * 3 + 1]
        self.trigram_firsts = trigram_choices[:n_trigrams]
        self.trigram_seconds = trigram_choices[n_trigrams : 2 * n_trigrams]
        self.trigram_thirds = trigram_choices[2 * n_trigrams : 3 * n_trigrams]

        # Create list of the tokens not selected from vocab, to be used as fillers/random tokens
        all_indices = set(range(1, vocab_size + 1))
        selected_indices = set(trigram_choices.tolist())
        self.filler_tokens = torch.tensor(list(all_indices - selected_indices))

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Pregenerate tokens tensor; -1 for BOS
        tensor_size = batch_size * (self.seq_len - 1)
        rand_indices = torch.randint(0, len(self.filler_tokens), (tensor_size,))
        tokens = self.filler_tokens[rand_indices].reshape((batch_size, self.seq_len - 1))

        # Choose trigrams from the list of trigrams randomly
        chosen_trigrams = torch.randint(0, self.n_trigrams, (batch_size,))
        batch_first_tokens = self.trigram_firsts[chosen_trigrams]
        batch_second_tokens = self.trigram_seconds[chosen_trigrams]
        batch_third_tokens = self.trigram_thirds[chosen_trigrams]

        # Skip distances should be distance from the end, not from the start.
        # Skip distance 0 should skip 0 tokens, i.e. AB(C); A thus indexed by seq[-2]
        skip_distances = torch.randint(
            (-1 * self.max_skip_distance) - 2, (-1 * self.min_skip_distance) - 2, (batch_size,)
        )

        # Replace random tokens with trigram tokens
        tokens[:, -1] = batch_second_tokens
        row_indices = torch.arange(len(skip_distances))
        tokens[row_indices, skip_distances] = batch_first_tokens

        # Insert BOS token
        tokens = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.long),  # BOS token
                tokens,
            ),
            dim=1,
        )

        return tokens, batch_third_tokens


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
