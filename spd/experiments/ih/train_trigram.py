from spd.utils.data_utils import TrigramDataset

# import torch
# from jaxtyping import Float
# from torch import Tensor
# from torch.utils.data import DataLoader, Dataset
# from spd.experiments.ih.model import InductionModelConfig, InductionTransformer
# from spd.log import logger
# from spd.utils.general_utils import set_seed


dataset = TrigramDataset(
    20, 15, "cpu", n_trigrams=3, min_skip_distance=1, max_skip_distance=2, size=100
)

batch = dataset.generate_batch(batch_size=20)
print(batch[1])
print(dataset.trigram_first, dataset.trigram_second, dataset.trigram_third)
