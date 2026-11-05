# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reject invalid configuration types before arithmetic or finite checks."""

import pytest

from openviking.integrations.langchain.ov_context import BudgetSpec, OVContextConfig


@pytest.mark.parametrize("field", ["commit_wait_timeout", "poll_interval", "poll_max_interval"])
@pytest.mark.parametrize("value", ["300", None, True, False, float("nan"), float("inf"), 0, -1])
def test_timeout_errors_identify_the_invalid_field(field, value):
    with pytest.raises(ValueError, match=field):
        OVContextConfig(**{field: value})


@pytest.mark.parametrize("factory", [OVContextConfig, BudgetSpec])
@pytest.mark.parametrize("field", ["base_context_ratio", "capacity_ratio"])
@pytest.mark.parametrize("value", ["0.9", None, True, float("nan"), float("inf")])
def test_ratio_types_are_rejected_consistently(factory, field, value):
    with pytest.raises(ValueError, match=field):
        factory(context_window=256000, output_reserve=8192, **{field: value})


@pytest.mark.parametrize("field", ["output_reserve", "safety_margin"])
@pytest.mark.parametrize("value", ["512", None, True, 1.5, -1])
def test_reserve_errors_identify_the_invalid_field(field, value):
    with pytest.raises(ValueError, match=field):
        OVContextConfig(**{field: value})


@pytest.mark.parametrize("factory", [OVContextConfig, BudgetSpec])
@pytest.mark.parametrize("ratios", [(0.9, 0.3), (-0.1, 0.9), (0.3, 1)])
def test_numeric_ratio_values_still_require_ordered_bounds(factory, ratios):
    with pytest.raises(ValueError, match="Require 0 <"):
        factory(
            context_window=256000,
            output_reserve=8192,
            base_context_ratio=ratios[0],
            capacity_ratio=ratios[1],
        )


def test_valid_integer_timeouts_and_zero_reserves_are_preserved():
    config = OVContextConfig(
        context_window=256000,
        commit_wait_timeout=300,
        poll_interval=1,
        poll_max_interval=2,
        output_reserve=0,
        safety_margin=0,
    )
    budget = BudgetSpec(
        config.context_window,
        config.output_reserve,
        config.safety_margin,
        base_context_ratio=config.base_context_ratio,
        capacity_ratio=config.capacity_ratio,
    )
    assert config.commit_wait_timeout == 300
    assert budget.input_limit == 256000
    assert budget.base_context_limit == 76800
    assert budget.capacity_limit == 230400
