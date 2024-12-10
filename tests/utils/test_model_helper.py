import pytest
import torch
from botorch.models.gp_regression import SingleTaskGP
from tests.helpers import (
    single_task_fixed_noise_m_1,
    single_task_fixed_noise_m_2,
    single_task_homo_noise_m_1,
    single_task_homo_noise_m_1_b3,
    single_task_homo_noise_m_2,
    single_task_homo_noise_m_2_b3,
)

from axtreme.utils.model_helpers import get_target_noise_singletaskgp, get_training_data_singletaskgp


@pytest.mark.parametrize(
    "model, expected_shape",
    [
        (single_task_homo_noise_m_1(), torch.Size([10, 1])),
        (single_task_homo_noise_m_1_b3(), torch.Size([10, 1])),
        (single_task_homo_noise_m_2(), torch.Size([10, 1])),
        (single_task_homo_noise_m_2_b3(), torch.Size([10, 1])),
    ],
)
def test_get_training_data_singletaskgp(model: SingleTaskGP, expected_shape: torch.Size):
    """Test training data of consistent format can be extracted from a model.

    Varient that effect the underling shape (b is batch shape, m is number of targets):
     - b = 0, m = 1
     - b = 3, m = 1
     - b = 0, m = 2
     - b = 3, m = 2

    Functionality is invariate to noise type, so that is not tested.

    NOTE: n0n=10 is dependant on the training data used in the model. If this changes the test will break.
    """
    result_shape = get_training_data_singletaskgp(model).shape

    torch.testing.assert_close(result_shape, expected_shape)


@pytest.mark.parametrize(
    "model, expected_shape",
    [
        # Homoskedastic
        (single_task_homo_noise_m_1(), torch.Size([1, 1])),
        (single_task_homo_noise_m_1_b3(), torch.Size([3, 1, 1])),
        (single_task_homo_noise_m_2(), torch.Size([1, 2])),
        (single_task_homo_noise_m_2_b3(), torch.Size([3, 1, 2])),
        # Fixed noise
        (single_task_fixed_noise_m_1(), torch.Size([10, 1])),
        (single_task_fixed_noise_m_2(), torch.Size([10, 2])),
    ],
)
def test_get_target_noise(model: SingleTaskGP, expected_shape: torch.Size):
    """Tests that different noise and shape types are treated constitenly are treated consistently."""
    result_shape = get_target_noise_singletaskgp(model).shape

    torch.testing.assert_close(result_shape, expected_shape)
