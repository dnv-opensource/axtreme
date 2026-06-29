import copy
import json
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
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.encoders import metric_to_dict
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
from axtreme.utils.serialization import _NotDeserialized


class TestQoIMetric:
    @pytest.mark.parametrize(
        "trial_idx, expected_mean",
        [
            # Trial without enough data
            # NOTE: QoIMetric.minimum_data_points is inclusive, and trial indexing is base 0.
            # So trial 1 has 2 data points.
            (1, float("nan")),
            # Trial 2 should have access to 3 data points, even thought 5 are in the experiment
            (2, 3.0),
            # all available data
            (4, 5.0),
        ],
    )
    def test_basic(self, trial_idx: int, expected_mean: float):
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

    def test_ignore_tracking_metrics(self):
        """This is very similar to test_basic, except we add an additional extraneous tracking metric that should
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

    def test_attaches_transforms(self):
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

    def test_serialization_manual(self):
        """Test that QoIMetric can be serialized.

        NOTE: This test will need to be updated if any change to `serialize_init_args` or `deserialize_init_args` is made.
        """

        class DummyQOIEstimator(QoIEstimator):
            def __init__(self) -> None:
                pass

        metric = QoIMetric(
            name="QoIMetric",
            qoi_estimator=DummyQOIEstimator(),  # pyright: ignore[reportAbstractUsage]
            minimum_data_points=3,
        )

        qoi_serialized = QoIMetric.serialize_init_args(metric)

        expected_serialization = {
            "name": "QoIMetric",
            "qoi_estimator": {"__type": "DummyQOIEstimator", "class_name": "DummyQOIEstimator"},
            "minimum_data_points": 3,
            "attach_transforms": False,
            "lower_is_better": None,
            "properties": {},
        }

        assert qoi_serialized == expected_serialization

    def test_deserialization_manual(self):
        """Test that QoIMetric can be deserialized.

        NOTE: This test will need to be updated if any change to `serialize_init_args` or `deserialize_init_args` is made.
        """

        serialized_metric = {
            "__type": "QoIMetric",
            "name": "QoIMetric",
            "qoi_estimator": {"__type": "DummyQOIEstimator", "class_name": "DummyQOIEstimator"},
            "minimum_data_points": 3,
            "attach_transforms": False,
            "lower_is_better": None,
            "properties": {},
        }

        _ = serialized_metric.pop("__type", None)  # Remove the __type key to simulate what object_from_json does
        deserialized_args = QoIMetric.deserialize_init_args(serialized_metric)
        deserialized_metric = QoIMetric(**deserialized_args)

        # Check that the deserialized runner has the same evaluation function class name
        assert isinstance(deserialized_metric.qoi_estimator, _NotDeserialized)
        assert object.__getattribute__(deserialized_metric.qoi_estimator, "class_name") == "DummyQOIEstimator"

    @pytest.mark.integration
    def test_serialize_to_and_from_json(self):
        """Check successful integration the the function the typically drive this process."""

        # Dummy object for test
        class DummyQOIEstimator(QoIEstimator):
            def __init__(self) -> None:
                pass

        metric = QoIMetric(
            name="QoIMetric",
            qoi_estimator=DummyQOIEstimator(),  # pyright: ignore[reportAbstractUsage]
            minimum_data_points=3,
        )

        encoder = {QoIMetric: metric_to_dict}
        decoder = {"QoIMetric": QoIMetric}

        metric_as_json = object_to_json(metric, encoder_registry=encoder)
        try:
            json.dumps(metric_as_json)  # Check if the serialized output is valid JSON
        except TypeError as e:
            pytest.fail(f"Serialized metric is not valid JSON: {e}")

        metric_deserialized = object_from_json(metric_as_json, decoder_registry=decoder)

        assert isinstance(metric_deserialized.qoi_estimator, _NotDeserialized)
        assert metric_deserialized.qoi_estimator.class_name == "DummyQOIEstimator"

    ##################################################
    ################# System test ####################
    ##################################################

    def test_system_test(self):
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
        - This is test is somewhat fragile, but is considered necessary as scoring is a critical part of the
        system and will cost a lot of time if incorrect.
        - None of the `examples` are used to provide an experiment etc. This is done so this test is not
        effected by changes to those files.

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

        # key check that the results are very similar. We allow a minor tolerance to account for the
        # slightly different GP hyperparams that can occur. We set `rtol` to be large so its ignored.
        np.testing.assert_allclose(
            baseline_results[minimum_data_points - 1 :, :],
            qoi_result[minimum_data_points - 1 :, :],
            atol=0.001,
            rtol=10,
        )
