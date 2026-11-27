# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-based input optimization through TabICLv2 and KumoTabular.

Predictions of a fitted model are differentiable with respect to the query
features, so query points can be moved by gradient descent on the predicted
value. Saving the plot requires ``matplotlib``.
"""

import matplotlib.pyplot as plt
import torch
from torch import Tensor

import sdm

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.empty(256, 1, device=device).uniform_(-3, 3)
y = (3 * x).sin() + 0.3 * x.square() + 0.05 * torch.randn_like(x)

grid = torch.linspace(-3, 3, 300, device=device)[:, None]
truth = ((3 * grid).sin() + 0.3 * grid.square()).squeeze(-1)
is_minimum = (truth[1:-1] < truth[:-2]) & (truth[1:-1] < truth[2:])
minima = grid[1:-1][is_minimum].squeeze(-1)
print(f"True minima: x={[round(v, 2) for v in minima.tolist()]}")


def mean_and_std(out: sdm.TableTensor) -> tuple[Tensor, Tensor]:
    quantiles = out.numerical  # [R, 999] at levels 0.001, ..., 0.999.
    # One standard deviation of a normal lies at the 15.9% and 84.1% levels:
    return quantiles.mean(dim=-1), (quantiles[:, 840] - quantiles[:, 158]) / 2


models = {
    "TabICLv2": sdm.models.TabICLv2(device=device),
    "KumoTabular (small)": sdm.models.KumoTabular(
        task="regression",
        size="small",
        device=device,
    ),
}
num_steps = 100

fig, axes = plt.subplots(1, len(models), figsize=(12, 4.5), sharey=True)
for ax, (name, model) in zip(axes, models.items(), strict=True):
    model.fit(x, y, num_estimators=8)
    with torch.inference_mode():
        mean, std = mean_and_std(model.predict(grid))

    # Query rows are predicted independently, so all start points move at once:
    query = torch.tensor([[-2.0], [0.3], [0.8]], device=device)
    query.requires_grad_()
    optimizer = torch.optim.Adam([query], lr=0.05)
    path_x: list[Tensor] = []
    path_y: list[Tensor] = []
    for step in range(num_steps + 1):
        optimizer.zero_grad()
        pred, _ = mean_and_std(model.predict(query))
        path_x.append(query.detach().squeeze(-1).clone())
        path_y.append(pred.detach())
        if step < num_steps:
            pred.sum().backward()
            optimizer.step()
            # Gradients vanish outside the context range.
            with torch.no_grad():
                query.clamp_(-3, 3)

    xs = torch.stack(path_x).cpu()  # [num_steps + 1, 3]
    ys = torch.stack(path_y).cpu()  # [num_steps + 1, 3]
    print(
        f"{name}: x={[round(v, 2) for v in xs[0].tolist()]} -> "
        f"{[round(v, 2) for v in xs[-1].tolist()]}, "
        f"predicted y={[round(v, 2) for v in ys[-1].tolist()]}"
    )

    grid_x, mean, std = grid.squeeze(-1).cpu(), mean.cpu(), std.cpu()
    ax.fill_between(grid_x, mean - std, mean + std, alpha=0.25, label="±1 std")
    ax.plot(grid_x, mean, label="predicted mean")
    ax.scatter(x.cpu(), y.cpu(), s=4, alpha=0.3, color="grey", label="context")
    ax.plot(xs, ys, color="tomato", lw=1.5)
    ax.plot(xs[0], ys[0], "o", color="tomato", mfc="white", label="start")
    ax.plot(xs[-1], ys[-1], "o", color="tomato", label="end")
    ax.set_title(name)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("optimize_inputs.png", dpi=120)
print("Saved optimize_inputs.png")
