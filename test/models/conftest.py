# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal

import pytest
import torch
from torch import Tensor

from sdm.models import ICLModel

Task = Literal["classification", "regression"]
QueryGradientCheck = Callable[[ICLModel, Task], None]


def _grad_task(
    task: Task,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    x_context = torch.randn(256, 8, dtype=dtype)
    if task == "regression":
        y_context = x_context[:, :1].sin() + x_context[:, 1:2]
    else:
        y_context = (x_context[:, :1] > 0).long()
    return x_context, y_context, torch.randn(6, 8, dtype=dtype)


def _loss_weight(out: Tensor, task: Task) -> Tensor:
    # Weight rows only for regression, where quantiles are sorted, and
    # classes for classification, whose probabilities sum to one.
    return torch.randn(
        out.size(0),
        1 if task == "regression" else out.size(-1),
        dtype=out.dtype,
    )


def _query_grad(model: ICLModel, query: Tensor, task: Task) -> Tensor:
    query = query.clone().requires_grad_()
    out = model.predict(query)
    weight = _loss_weight(out.numerical, task)
    (grad,) = torch.autograd.grad((out.numerical * weight).sum(), query)
    return grad


@pytest.fixture
def check_query_gradients() -> QueryGradientCheck:
    def check(model: ICLModel, task: Task) -> None:
        x_context, y_context, x_query = _grad_task(task)
        # More than 4 estimators share each `Choice` option between members.
        num_estimators = 8
        model.fit(
            x_context,
            y_context,
            num_estimators=num_estimators,
            generator=torch.Generator().manual_seed(0),
        )
        expected = model.predict(x_query)
        assert torch.is_inference(expected)

        query = x_query.clone().requires_grad_()
        out = model.predict(query)
        weight = _loss_weight(out.numerical, task)
        (grad,) = torch.autograd.grad((out.numerical * weight).sum(), query)

        torch.testing.assert_close(
            out.numerical,
            expected.numerical,
            atol=1e-4,
            rtol=1e-4,
        )
        assert grad.isfinite().all()
        assert grad.count_nonzero() > 0

        # The uncached forward recomputes the context with the same estimators.
        query = x_query.clone().requires_grad_()
        out = model(
            x_context,
            y_context,
            query,
            num_estimators=num_estimators,
            generator=torch.Generator().manual_seed(0),
        )
        (expected_grad,) = torch.autograd.grad(
            (out.numerical * weight).sum(), query
        )
        torch.testing.assert_close(grad, expected_grad, atol=1e-5, rtol=1e-4)

    return check


@pytest.fixture
def check_query_finite_differences() -> QueryGradientCheck:
    def check(model: ICLModel, task: Task) -> None:
        model = model.to(torch.float64)
        x_context, y_context, query = _grad_task(task, torch.float64)
        model.fit(x_context, y_context, num_estimators=8)

        query.requires_grad_()
        out = model.predict(query)
        weight = _loss_weight(out.numerical, task)
        (grad,) = torch.autograd.grad((out.numerical * weight).sum(), query)

        eps = 1e-4
        for _ in range(3):
            direction = torch.randn_like(query)
            direction /= direction.norm()
            upper = model.predict(query.detach() + eps * direction).numerical
            lower = model.predict(query.detach() - eps * direction).numerical
            finite_difference = ((upper - lower) * weight).sum() / (2 * eps)
            torch.testing.assert_close(
                (grad * direction).sum(),
                finite_difference,
                atol=1e-7,
                rtol=1e-4,
            )

    return check


@pytest.fixture
def check_query_gradients_at_edges() -> QueryGradientCheck:
    def check(model: ICLModel, task: Task) -> None:
        x_context, y_context, x_query = _grad_task(task)
        # Zero-inflated, with ties at its minimum.
        x_context[:, 0].clamp_(min=0)
        model.fit(x_context, y_context, num_estimators=8)

        below_minimum = x_query.clone()
        below_minimum[:, 0] = -1.0
        assert _query_grad(model, below_minimum, task).isfinite().all()

        missing = x_query.clone()
        missing[::2, 1] = torch.nan
        grad = _query_grad(model, missing, task)
        assert grad.isfinite().all()
        assert (grad[::2, 1] == 0).all()

    return check
