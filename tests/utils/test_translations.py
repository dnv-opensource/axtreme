import numpy as np
import torch
from ax import ParameterType, RangeParameter, SearchSpace
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.modelbridge.modelbridge_utils import (
    array_to_observation_data,
    observation_data_to_array,
    observation_features_to_array,
)
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.unit_x import UnitX
from numpy.testing import assert_allclose

from axtreme.utils import transforms


# TODO(sw): Come back and set this up for a multi input and output to ake sure the shapes are right
def test_translate_standardisey():
    ob_datas = array_to_observation_data(
        f=np.array(
            [
                [1, 7],
                [2, 6],
            ]
        ),
        cov=np.array(
            [
                [
                    [0.5, 0.1],
                    [0.1, 0.5],
                ],
                [
                    [1, 0.1],
                    [0.1, 2],
                ],
            ]
        ),
        outcomes=["a", "b"],
    )
    obs = [Observation(features=ObservationFeatures(None), data=obs_data) for obs_data in ob_datas]

    # Standardises the y for each metric
    transform = StandardizeY(observations=obs)

    input_transform, output_transform = transforms.translate_standardisey(transform, col_names=["a", "b"])
    assert input_transform is None
    assert output_transform is not None

    # we care if the transform works, not the internals, so we test the output
    ### Set up the data in model space to be untransformed
    # data for ax
    obs_model_space = Observation(
        features=ObservationFeatures(None),  # Note used, put none to make this obvious
        data=ObservationData(
            metric_names=["a", "b"],
            means=np.array([3, 8]),
            covariance=np.array(
                [
                    [1, 0.1],
                    [0.1, 2],
                ],
            ),
        ),
    )

    # same data for botorch
    y, yvar = observation_data_to_array(outcomes=["a", "b"], observation_data=[obs_model_space.data])
    y = torch.tensor(y)
    yvar = torch.tensor(yvar)

    ### Transform each to the problem space
    # ax
    obs_data_problem_space_expected = transform.untransform_observations([obs_model_space])[0].data

    # Make the equivalent model space value in tensor form
    botorch_output_means, botorch_output_cov = output_transform.untransform(Y=y, Yvar=yvar)
    assert botorch_output_cov is not None

    assert_allclose(obs_data_problem_space_expected.means, botorch_output_means[0].numpy())
    assert_allclose(obs_data_problem_space_expected.covariance, botorch_output_cov[0].numpy())


def test_translate_unitx():
    junk_search_space = SearchSpace(
        parameters=[
            RangeParameter(name="a", parameter_type=ParameterType.FLOAT, lower=0, upper=50),
            RangeParameter(name="b", parameter_type=ParameterType.FLOAT, lower=0, upper=25),
        ]
    )

    transform = UnitX(junk_search_space)

    input_transform, output_transform = transforms.translate_unitx(transform)
    assert input_transform is not None
    assert output_transform is None

    #### Create the feature to be transofrmed
    feature_ax = ObservationFeatures(
        parameters={
            "a": 25,
            "b": 12.5,
        }
    )
    feature_botorch = torch.tensor(observation_features_to_array(["a", "b"], [feature_ax]))

    ### Transform the features
    ax_transformed = transform.transform_observation_features([feature_ax])
    botorch_transformed = input_transform.transform(feature_botorch)

    ### Check values are equivalent
    assert_allclose(observation_features_to_array(["a", "b"], ax_transformed), botorch_transformed.numpy())
