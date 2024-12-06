# Pytest
Provides and overview of the pytest design decisions made.


## Naming:
Function names should be very explicit about what they test **within the context of their location**. For example:
- function `test_forward_shape()` is an acceptable when it is located at a path such as this `tests/sampling/test_mean_sampler.py` because pytest will produce the following information on failure. `FAILED tests/sampling/test_mean_sampler.py::test_forward[posterior_1point_2Doutput]`.

## Question and Answer:
Design questions we had and the answer we arrived at

### Issue: how to combine fixture with expected output
We have different posterior for different types of GPs, sometimes we want to test"
- specific posterior produces a specific result.
- All posterior work (test is invariant or has generic results)

Idea case
- All individual posterior are fixtures. ( can then be used with paramterise as explained [here](https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/).
- Then have a posterior object that just puts all those fixtures into itself with `pytext.fixture(param = [posterior1,...])`
- Pros:
    - Don't need to import anywhere
    - Can run directly on the default set with minimal code
- Cons:
    - Doesn't work by default, might be able to implement
        - Likely hard to implement because cant put fixture in decorators - both are constructed at collection time.


#### Possible solutions:
- Alternative 1:
    o Individual things are functions:
        o Import them in the files need them, can put firstly into paramterise
    o Can then make a fixture with all of them easilyt
    o PROS:
        o Parametrize is simple
        o Running on all is simple
    o Cons:
        o Can't just refere to a single posterior in the arg
            - Don't expect to many of these cases,
            - If this is a real paic, could make parametrised wrappers for

#### Decision
go with Alternative 1 for now.

### Issues: why use fixtures over function.
What is the benefit of fixtures, can't we just call function? (same q at [stackoverflow](https://stackoverflow.com/questions/62376252/when-to-use-pytest-fixtures))
- reasons to go with a fixture
	- Things that take a long time - can scope them with a fixture to avoid repeat set up
	- Don't have to explicitly import: can just refer to fixture by name, found at collection time.
	- Parameterised fixtures get each variant run automatically
#### Decision
	- Doesn't matter too much either way - you can also mix them.
	- If doesn't effect you too much, go with fixture, follows their patern
