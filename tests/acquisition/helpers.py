import torch
from botorch.models.model import Model


def check_same_model_data(model_actual: Model, model_expected: Model):
    torch.testing.assert_close(model_actual.train_inputs[0], model_expected.train_inputs[0])
    torch.testing.assert_close(model_actual.train_targets, model_expected.train_targets)
    torch.testing.assert_close(model_actual.likelihood.noise_covar.noise, model_expected.likelihood.noise_covar.noise)


def check_posterior(model_actual: Model, model_expected: Model):
    torch.testing.assert_close(model_actual.distribution.mean, model_expected.distribution.mean)
    torch.testing.assert_close(
        model_actual.distribution.covariance_matrix, model_expected.distribution.covariance_matrix
    )
