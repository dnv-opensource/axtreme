import copy
import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from ax import Data, Metric, Models
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.metric import MetricFetchResult
from ax.utils.common.result import Ok
from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_multi_objective_optimization_config
from botorch.models.model import Model
from torch.utils.data import DataLoader

from axtreme import qoi
from axtreme.data import MinimalDataset
from axtreme.experiment import add_sobol_points_to_experiment
from axtreme.metrics import QoIMetric
from axtreme.qoi import MarginalCDFExtrapolation, QoIEstimator
from axtreme.sampling.ut_sampler import UTSampler


@pytest.mark.parametrize(
    "trial_idx, expected_mean",
    [
        # Trial without enough data
        # NOTE: QoIMetric.minimum_data_points is inclusive, and trial indexing is base 0. So trial 1 has 2 data points.
        (1, float("nan")),
        # Trial 2 should have access to 3 data points, even thought 5 are in the experiment
        (2, 3.0),
        # all available data
        (4, 5.0),
    ],
)
def test_qoimetric_basic(trial_idx: int, expected_mean: float):
    """Check that the internal GP is trained with the appropriate number of data points.

    This is done by using a mock QoIEstimator that returns the number of training points in the GP
    """

    # Setup
    class DummyQOIEstimator(QoIEstimator):
        """A simple QOIEstimator that just returns the number of training points."""

        def __call__(self, model: Model) -> torch.Tensor:
            # Note the shape is always (n,d) regardless of number of targets etc
            number_training_points = model.train_inputs[0].shape[-2]
            # Duplicate this value so downstream variance calculation can still run
            return torch.tensor([number_training_points] * 5, dtype=torch.float)

    exp = get_branin_experiment(minimize=True)
    add_sobol_points_to_experiment(exp, 5)

    metric = QoIMetric(
        name="QoIMetric",
        qoi_estimator=DummyQOIEstimator(),
        minimum_data_points=3,
    )

    trial = exp.trials[trial_idx]

    # Perform test
    results = metric.fetch_trial_data(trial)

    # Check results
    data = cast("Data", results.value)
    data_dict = data.df.loc[0].to_dict()

    assert data_dict["metric_name"] == "QoIMetric"
    assert data_dict["trial_index"] == trial_idx
    assert data_dict["mean"] == expected_mean or (math.isnan(data_dict["mean"]) and math.isnan(expected_mean))


def test_qoimetric_ignore_tracking_metrics():
    """This is very similar to test_qoimetric_basic, except we add an additional extraneous tracking metric that should
    be ignored.
    """

    # This metric does not do anything, it is just here to test that the QoIMetric ignores it
    class ExtraneousMetric(Metric):
        def fetch_trial_data(
            self,
            trial: BaseTrial,
            **kwargs: Any,  # noqa: ANN401 NOTE: kwargs is needed to match the signature of the parent class.
        ) -> MetricFetchResult:
            output = {
                "metric_name": "ExtraneousMetric",
                "trial_index": trial.index,
                "mean": 0.0,
                "sem": 0.0,
                "metric_value": 0.0,
            }
            return Ok(value=Data(df=pd.DataFrame([output])))

    class IgnoreTrackingMetrics(QoIEstimator):
        """Checks tracking metrics have been ignored by checking the training targets of the model passed."""

        def __call__(self, model: Model) -> torch.Tensor:
            # 1 target; shape will be (n,). for 2 targets the shape will be (n,2)
            training_targets = model.train_targets
            # check only one target was passed:
            assert len(training_targets.shape)
            # Check the values are not 0
            assert (training_targets != 0).all()

            return torch.rand(5)

    exp = get_branin_experiment(minimize=True)
    _ = exp.add_tracking_metric(ExtraneousMetric(name="ExtraneousMetric"))
    add_sobol_points_to_experiment(exp, 5)

    metric = QoIMetric(
        name="QoIMetric",
        qoi_estimator=IgnoreTrackingMetrics(),
        minimum_data_points=3,
    )

    trial = exp.trials[4]

    # Perform test - assert happens within the IgnoreTrackingMetrics
    _ = metric.fetch_trial_data(trial)


def test_qoimetric_attaches_transforms():
    """This only checks that transforms are attached to QoIEstimator.

    Checking transforms are correct is covered in other unit tests.
    """

    class TransformChecked(QoIEstimator):
        """Checks `input_transform` and `outcome_transform` are attached"""

        def __call__(self, model: Model) -> torch.Tensor:
            assert hasattr(self, "input_transform")
            assert hasattr(self, "outcome_transform")
            return torch.rand(5)

    exp = get_branin_experiment(minimize=True)
    add_sobol_points_to_experiment(exp, 5)

    metric = QoIMetric(
        name="QoIMetric",
        qoi_estimator=TransformChecked(),
        minimum_data_points=3,
        attach_transforms=True,
    )
    # select the a trail that meets minimum data requirement
    trial = exp.trials[2]

    # Perform test - assert happens within the TransformChecked
    _ = metric.fetch_trial_data(trial)


##################################################
################# System test ####################
##################################################


def test_qoimetric_system_test():
    """We perform a system test integrating qoi tracking metrics with a realistic QoI .

    Compares using a tracking metric to the more manual approach used thus far to ensure consistency.

    Test overview:
    - Set up exp using ax helpers (one for baseline approach and one for tracking metric)
    - Define the points that will be used to run the experiment
    - set up the qoi
    - run the basline approach.
    - run the tracking metrics approach.
    - Check the QoI calculation outputs (mean, sem) are very similar.

    Notes:
    - This is test is somewhat fragile, but is considered necessary as scoring is a critical part of the system and will
    cost a lot of time if incorrect.
    - None of the `examples` are used to provide an experiment etc. This is done so this test is not effected by changes
    to those files.

    """

    # Build an experiment with multi-objective optimisation.
    # This produce an experiment with 2 objective, both identical
    exp = get_branin_experiment(minimize=True)
    exp.optimization_config = get_branin_multi_objective_optimization_config()

    # create an identical experiment where the QoI will be calculated manually at each step
    exp_baseline = copy.deepcopy(exp)

    # 7 points within the search space (bounds are x1_range=[-5.0, 10.0] x2_range=[0.0, 15.0])
    points = torch.tensor(
        [
            [9.9773, 4.5281],
            [0.3684, 13.8791],
            [-0.4806, 12.5667],
            [8.7557, 12.6992],
            [-2.5663, 1.1046],
            [-2.3586, 3.4567],
            [-3.3876, 11.2420],
        ]
    )

    # Generate a fixed dataset in the ares:
    rng = np.random.default_rng(seed=42)
    sample = rng.random((500, 2))
    x_mins = np.array([-5.0, 0])
    x_maxes = np.array([10.0, 15])
    env_data = x_mins + (x_maxes - x_mins) * sample

    # set up a QoIEstimator.
    dataset = MinimalDataset(env_data)
    dataloader = DataLoader(dataset, batch_size=256)

    posterior_sampler = UTSampler()

    qoi_estimator = MarginalCDFExtrapolation(
        # random dataloader give different env samples for each instance
        env_iterable=dataloader,
        period_len=1_000,  # we can set this to whatever we want
        quantile=torch.tensor(0.5),
        quantile_accuracy=torch.tensor(0.01),
        # IndexSampler needs to be used with GenericDeterministicModel. Each sample just selects the mean.
        posterior_sampler=posterior_sampler,
    )

    ############ Calculate the QOI on an experiment
    minimum_data_points = 3

    ######## Run the baseline approach #########
    means = []
    sems = []
    for idx, point in enumerate(points):
        params = {"x1": float(point[0]), "x2": float(point[1])}
        arm = Arm(parameters=params)
        trial = exp_baseline.new_trial()
        _ = trial.add_arm(arm)
        _ = trial.run()
        _ = trial.mark_completed()

        if idx + 1 < minimum_data_points:
            means.append(float("nan"))
            sems.append(float("nan"))
        else:
            botorch_model_bridge = Models.BOTORCH_MODULAR(
                experiment=exp_baseline,
                data=exp_baseline.fetch_data(),
            )

            qoi_estimator = qoi.utils.attach_transforms_to_qoi_estimator(botorch_model_bridge, qoi_estimator)

            model = botorch_model_bridge.model.surrogate.model
            estimates = qoi_estimator(model)
            means.append(float(qoi_estimator.mean(estimates)))
            sems.append(float(qoi_estimator.var(estimates)) ** 0.5)

    df_baseline = pd.DataFrame({"mean": means, "sem": sems})

    ######## Using tracking metrics #########

    qoi_metric = QoIMetric(
        name="QoIMetric",
        qoi_estimator=qoi_estimator,
        minimum_data_points=minimum_data_points,
        attach_transforms=True,
    )
    _ = exp.add_tracking_metric(qoi_metric)

    for point in points:
        params = {"x1": float(point[0]), "x2": float(point[1])}
        arm = Arm(parameters=params)
        trial = exp.new_trial()
        _ = trial.add_arm(arm)
        _ = trial.run()
        _ = trial.mark_completed()

    df_tracking_metrics = exp.fetch_data().df

    df_qoi = df_tracking_metrics.loc[df_tracking_metrics["metric_name"] == "QoIMetric", ["mean", "sem"]]

    qoi_result = df_qoi.to_numpy()
    baseline_results = df_baseline.to_numpy()

    # minor check that nan appear in the expected place
    assert np.isnan(baseline_results[: minimum_data_points - 1, :]).all()
    assert np.isnan(qoi_result[: minimum_data_points - 1, :]).all()

    # key check that the results are very similar. We allow a minor tolerance to account for the slightly different GP
    # hyperparams that can occur. We set `rtol` to be large so its ignored.
    np.testing.assert_allclose(
        baseline_results[minimum_data_points - 1 :, :], qoi_result[minimum_data_points - 1 :, :], atol=0.001, rtol=10
    )
