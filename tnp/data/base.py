import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class BaseBatch(ABC):
    pass


@dataclass
class ImageBatch(BaseBatch):
    y_grid: torch.Tensor
    mc_grid: torch.Tensor
    mt_grid: torch.Tensor
    yt: torch.Tensor


@dataclass
class Batch(BaseBatch):
    x: torch.Tensor
    y: torch.Tensor

    xt: torch.Tensor
    yt: torch.Tensor

    xc: torch.Tensor
    yc: torch.Tensor


class GroundTruthPredictor(ABC):
    def __init__(self):
        pass

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Any:
        raise NotImplementedError

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class DataGenerator(torch.utils.data.IterableDataset, ABC):
    def __init__(
        self,
        *,
        samples_per_epoch: int,
        batch_size: int,
        deterministic: bool = False,
        deterministic_seed: int = 0,
        **kwargs,
    ):
        """Base data generator, which can be used to derive other data generators,
        such as synthetic generators or real data generators.

        Arguments:
            samples_per_epoch: Number of samples per epoch.
            batch_size: Batch size.
        """
        super().__init__(**kwargs)

        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.num_batches = samples_per_epoch // batch_size

        # Set batch counter.
        self.batch_counter = 0
        self.deterministic = deterministic
        self.deterministic_seed = deterministic_seed
        self.batches = None

    def __iter__(self):
        """Reset the batch counter and return this iterable dataset.

        For deterministic generators, batches are generated once from
        ``deterministic_seed`` and cached. The caller's Python, NumPy,
        PyTorch CPU, and initialized CUDA RNG states are restored exactly
        afterwards.
        """
        if self.deterministic and self.batches is None:
            torch_state = torch.get_rng_state()
            numpy_state = np.random.get_state()
            python_state = random.getstate()

            cuda_states = None
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                cuda_states = torch.cuda.get_rng_state_all()

            try:
                seed = int(self.deterministic_seed)

                # Seed the global CPU generator used by torch.rand,
                # torch.randn, torch.randint, and related operations.
                torch.random.default_generator.manual_seed(seed)

                # Only touch CUDA RNGs if CUDA was already initialized;
                # CPU data-loader workers should not initialize CUDA.
                if cuda_states is not None:
                    torch.cuda.manual_seed_all(seed)

                np.random.seed(seed)
                random.seed(seed)

                self.batches = [
                    self.generate_batch()
                    for _ in range(self.num_batches)
                ]

            finally:
                torch.set_rng_state(torch_state)
                np.random.set_state(numpy_state)
                random.setstate(python_state)

                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)

        self.batch_counter = 0
        return self

    def __next__(self) -> BaseBatch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """

        if self.batch_counter >= self.num_batches:
            raise StopIteration

        if self.deterministic and self.batches is not None:
            batch = self.batches[self.batch_counter]
        else:
            batch = self.generate_batch()

        self.batch_counter += 1
        return batch

    @abstractmethod
    def generate_batch(self) -> BaseBatch:
        """Generate batch of data.

        Returns:
            batch: Tuple of tensors containing the context and target data.
        """
