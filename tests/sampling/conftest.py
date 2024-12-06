"""Fixtures required for this subdirectory."""
# sourcery skip: dont-import-test-modules

import pytest
from tests.sampling import helpers


@pytest.fixture(
    params=[
        helpers.posterior_n1_t1(),
        helpers.posterior_n2_t1(),
        helpers.posterior_n1_t2(),
        helpers.posterior_n3_t2(),
        helpers.posterior_b2_n3_t2(),
    ],
    # provides naming which makes it easi
    ids=[
        "posterior_n1_t1",
        "posterior_n2_t1",
        "posterior_n1_t2",
        "posterior_n3_t2",
        "posterior_b2_n3_t2",
    ],
)
def posterior(request: pytest.FixtureRequest):
    """Provides a generic set of posteriors to be tested against.

    This is useful when functionality should be invariant to specific type of posterior.
    """
    return request.param
