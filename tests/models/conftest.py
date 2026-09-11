import copy

import pytest

from scenario_builders import BASE_SCENARIO


@pytest.fixture
def raw_scenario():
    return copy.deepcopy(BASE_SCENARIO)
