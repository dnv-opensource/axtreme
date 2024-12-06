from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from axtreme.eval.object_logging import NestedDict, unpack_object, unpack_object_str_content


@dataclass
class Level2:
    a: str = "blah"
    b: None = None
    c: tuple[int, ...] = (1, 2, 3)


@dataclass
class Level1:
    level2: Level2
    x: str = "x"


@pytest.fixture
def l1():
    l2 = Level2()
    return Level1(level2=l2)


def test_unpack_object_depth_0(l1: Level1):
    unpacked = unpack_object(l1, depth=0)
    assert unpacked == {"__class__": Level1}


def test_unpack_object_depth_1(l1: Level1):
    unpacked = unpack_object(l1, depth=1)

    # fmt: off
    expected_value = {
        "__class__": Level1,
        "level2": {
            "__class__": Level2
        },
        "x": "x"
    }
    # fmt: on

    assert unpacked == expected_value


def test_unpack_object_depth_2(l1: Level1):
    unpacked = unpack_object(l1, depth=2)

    # fmt: off
    expected_value = {
        '__class__': Level1,
        'level2': {
            '__class__': Level2,
            'a': 'blah',
            'b': None,
            'c': (1, 2, 3)
        },
        'x': 'x'
    }
    # fmt: on

    assert unpacked == expected_value


def test_unpack_object_custom_config_depth_0(l1: Level1):
    """Custom config will be found ar depth 0 and override the entire result."""
    config: dict[type, Callable[[Any], NestedDict]] = {Level1: lambda _: {"junk1": "junk1"}}
    unpacked = unpack_object(l1, custom_unpacking_config=config, depth=2)
    assert unpacked == {"junk1": "junk1"}


def test_unpack_object_custom_config_depth_1(l1: Level1):
    """Custom config will overide the portion of the result"""
    config: dict[type, Callable[[Any], NestedDict]] = {Level2: lambda _: {"junk1": "junk1"}}
    unpacked = unpack_object(l1, custom_unpacking_config=config, depth=2)

    # fmt: off
    expected_value = {
        "__class__": Level1,
        "level2": {
            "junk1": "junk1"
        },
        "x": "x"
    }
    # fmt: on
    assert unpacked == expected_value


def test_unpack_object_no__dict__method():
    unpacked = unpack_object([1, 2, 3], depth=1)

    assert unpacked == {"__class__": list}


def test_unpack_object_str_content(l1: Level1):
    unpacked = unpack_object_str_content(l1, depth=2)

    # fmt: off
    expected_value = {
        '__class__': str(Level1),
        'level2': {
            '__class__': str(Level2),
            'a': 'blah',
            'b': str(None),
            'c': str((1, 2, 3))
        },
        'x': 'x'
    }
    # fmt: on

    assert unpacked == expected_value


@pytest.mark.integration
def test_unpack_object_integration():
    """This is mainly to document/demonstrate its functionality on more complicated objects."""

    from collections.abc import Sized
    from typing import cast

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from axtreme.data import BatchInvariantSampler2d, MinimalDataset, SizableSequentialSampler
    from axtreme.qoi import GPBruteForce
    from axtreme.sampling import NormalIndependentSampler

    torch.set_default_dtype(torch.float64)

    N_ENV_SAMPLES_PER_PERIOD = 31  # noqa: N806

    dataset = MinimalDataset(np.arange(10_000).reshape(-1, 2))
    posterior_sampler = NormalIndependentSampler(torch.Size([50]))
    # gp brute force
    n_periods = 50
    sampler = SizableSequentialSampler(
        data_source=cast(Sized, dataset),
        num_samples=N_ENV_SAMPLES_PER_PERIOD * n_periods,
    )
    batch_sampler = BatchInvariantSampler2d(
        sampler=sampler,
        batch_shape=torch.Size([n_periods, 64]),
    )
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    qoi_estimator_gp_brute_force = GPBruteForce(
        env_iterable=dataloader,
        posterior_sampler=posterior_sampler,
        no_grad=True,
        seed=1234,
    )

    # here we simply check that it can run to completion. It will produce result similar to the ones below.
    # we don't check the actual results as this will change if underling methods add attributes etc.
    # That is too much overhead to keep this test up to date.
    _ = unpack_object_str_content(qoi_estimator_gp_brute_force, depth=2)
    # Expected output:
    # fmt: off
    _ = {
        "__class__": "<class 'axtreme.qoi.gp_bruteforce.GPBruteForce'>",
        "device": "cpu",
        "env_iterable": {
            "__class__": "<class 'torch.utils.data.dataloader.DataLoader'>",
            "batch_sampler": {
                "__class__": "<class 'axtreme.data.batch_invariant_sampler.BatchInvariantSampler2d'>"
            },
            "batch_size": "None",
            "collate_fn": {
                "__class__": "<class 'function'>"
            },
            "dataset": {
                "__class__": "<class 'axtreme.data.dataset.MinimalDataset'>"
            },
            "drop_last": "False",
            "generator": "None",
            "num_workers": "0",
            "persistent_workers": "False",
            "pin_memory": "False",
            "pin_memory_device": "",
            "prefetch_factor": "None",
            "sampler": {
                "__class__": "<class 'torch.utils.data.sampler.SequentialSampler'>"
            },
            "timeout": "0",
            "worker_init_fn": "None"
        },
        "erd_samples_per_period": "1",
        "input_transform": "None",
        "no_grad": "True",
        "outcome_transform": "None",
        "posterior_sampler": {
            "__class__": "<class 'axtreme.sampling.normal_independent_sampler.NormalIndependentSampler'>",
            "sample_shape": "torch.Size([50])",
            "seed": "332061",
            "training": "True"
        },
        "seed": "1234"
    }
    # fmt: on
