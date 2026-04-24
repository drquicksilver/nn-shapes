import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DOMAIN_MIN = -1.0
DOMAIN_MAX = 1.0


def label_circle(points: torch.Tensor, radius: float = 0.55) -> torch.Tensor:
    """Return 1 for points inside a circle centered at the origin."""
    squared_radius = points[:, 0].square() + points[:, 1].square()
    return (squared_radius <= radius * radius).float()


def label_box(
    points: torch.Tensor, half_width: float = 0.55, half_height: float = 0.35
) -> torch.Tensor:
    """Return 1 for points inside an axis-aligned box centered at the origin."""
    inside_x = points[:, 0].abs() <= half_width
    inside_y = points[:, 1].abs() <= half_height
    return (inside_x & inside_y).float()


SHAPES = {
    "circle": label_circle,
    "box": label_box,
}


class ShapeClassifier(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()
        layers = []
        input_size = 2

        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points).squeeze(-1)


@dataclass(frozen=True)
class TrainingResult:
    final_loss: float
    final_accuracy: float


def choose_seed(seed: Optional[int]) -> int:
    if seed is not None:
        return seed
    return random.SystemRandom().randint(0, 2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def sample_points(num_samples: int) -> torch.Tensor:
    points = torch.rand(num_samples, 2)
    return points * (DOMAIN_MAX - DOMAIN_MIN) + DOMAIN_MIN


def make_dataset(shape: str, num_samples: int) -> TensorDataset:
    points = sample_points(num_samples)
    labels = SHAPES[shape](points)
    return TensorDataset(points, labels)


def train_model(
    model: ShapeClassifier,
    dataset: TensorDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> TrainingResult:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_items = 0

        for points, labels in loader:
            logits = model(points)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * points.size(0)
            total_items += points.size(0)

        final_loss = total_loss / total_items
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}  loss {final_loss:.4f}")

    return TrainingResult(
        final_loss=final_loss,
        final_accuracy=calculate_accuracy(model, dataset),
    )


def calculate_accuracy(model: ShapeClassifier, dataset: TensorDataset) -> float:
    points, labels = dataset.tensors
    with torch.no_grad():
        predicted_inside = torch.sigmoid(model(points)) >= 0.5
    correct = predicted_inside == labels.bool()
    return correct.float().mean().item()


def make_prediction_grid(
    model: ShapeClassifier, grid_size: int = 240
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, grid_size)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(grid_points)).reshape(grid_size, grid_size)

    return grid_x, grid_y, probabilities


def plot_decision_boundary(
    model: ShapeClassifier,
    dataset: TensorDataset,
    shape: str,
    seed: int,
    result: TrainingResult,
    output_path: Path,
    show: bool,
) -> None:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_x, grid_y, probabilities = make_prediction_grid(model)
    points, labels = dataset.tensors
    inside = labels.bool()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    background = ax.contourf(
        grid_x.numpy(),
        grid_y.numpy(),
        probabilities.numpy(),
        levels=20,
        cmap="RdYlBu_r",
        alpha=0.75,
    )
    ax.contour(
        grid_x.numpy(),
        grid_y.numpy(),
        probabilities.numpy(),
        levels=[0.5],
        colors="black",
        linewidths=2,
    )

    ax.scatter(
        points[~inside, 0].numpy(),
        points[~inside, 1].numpy(),
        s=8,
        c="#3b4252",
        alpha=0.35,
        label="outside samples",
    )
    ax.scatter(
        points[inside, 0].numpy(),
        points[inside, 1].numpy(),
        s=8,
        c="#d08770",
        alpha=0.45,
        label="inside samples",
    )

    ax.set_title(
        f"{shape} | seed {seed} | loss {result.final_loss:.4f} | "
        f"accuracy {result.final_accuracy:.3f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.legend(loc="upper right")
    fig.colorbar(background, ax=ax, label="predicted probability of inside")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small neural network to learn inside/outside a shape."
    )
    parser.add_argument("--shape", choices=SHAPES.keys(), default="circle")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/decision_boundary.png"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = choose_seed(args.seed)
    set_seed(seed)
    print(f"seed: {seed}")

    dataset = make_dataset(args.shape, args.num_samples)
    model = ShapeClassifier(
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
    )
    result = train_model(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(f"final accuracy: {result.final_accuracy:.3f}")

    plot_decision_boundary(
        model=model,
        dataset=dataset,
        shape=args.shape,
        seed=seed,
        result=result,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
