import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from main import (
    ShapeClassifier,
    label_box,
    label_circle,
    make_contour_colormap,
    make_epoch_output_path,
    train_model,
)


class ShapeLabelTests(unittest.TestCase):
    def test_circle_labels_obvious_inside_and_outside_points(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0],
                [0.25, 0.25],
                [0.8, 0.0],
                [0.0, -0.8],
            ]
        )

        labels = label_circle(points)

        torch.testing.assert_close(labels, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_box_labels_obvious_inside_and_outside_points(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.3],
                [0.7, 0.0],
                [0.0, -0.5],
            ]
        )

        labels = label_box(points)

        torch.testing.assert_close(labels, torch.tensor([1.0, 1.0, 0.0, 0.0]))


class ModelTests(unittest.TestCase):
    def test_model_returns_one_logit_per_point(self) -> None:
        model = ShapeClassifier(hidden_size=8, hidden_layers=1)
        points = torch.zeros(5, 2)

        logits = model(points)

        self.assertEqual(logits.shape, torch.Size([5]))


class TrainingSnapshotTests(unittest.TestCase):
    def test_train_model_calls_snapshot_callback_before_training_and_at_epoch_interval(
        self,
    ) -> None:
        model = ShapeClassifier(hidden_size=4, hidden_layers=1)
        dataset = TensorDataset(
            torch.zeros(4, 2),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
        )
        snapshot_epochs = []

        train_model(
            model=model,
            dataset=dataset,
            epochs=4,
            batch_size=2,
            learning_rate=0.001,
            snapshot_interval_epochs=2,
            snapshot_callback=lambda epoch, result: snapshot_epochs.append(epoch),
        )

        self.assertEqual(snapshot_epochs, [0, 2, 4])

    def test_train_model_rejects_non_positive_snapshot_epoch_interval(self) -> None:
        model = ShapeClassifier(hidden_size=4, hidden_layers=1)
        dataset = TensorDataset(torch.zeros(2, 2), torch.tensor([0.0, 1.0]))

        with self.assertRaisesRegex(ValueError, "must be positive"):
            train_model(
                model=model,
                dataset=dataset,
                epochs=1,
                batch_size=1,
                learning_rate=0.001,
                snapshot_interval_epochs=0,
                snapshot_callback=lambda epoch, result: None,
            )


class ContourColormapTests(unittest.TestCase):
    def test_contour_colormap_has_hard_change_at_half_probability(self) -> None:
        colormap = make_contour_colormap()

        below_half = colormap(0.49)[:3]
        at_half = colormap(0.50)[:3]
        above_half = colormap(0.51)[:3]

        self.assertGreater(below_half[2], below_half[1])
        self.assertGreater(at_half[0], at_half[1])
        self.assertGreater(above_half[0], above_half[1])


class OutputPathTests(unittest.TestCase):
    def test_make_epoch_output_path_pads_to_total_epoch_digits(self) -> None:
        output_path = make_epoch_output_path(Path("outputs/circle.png"), 25, 500)

        self.assertEqual(output_path, Path("outputs/circle_epoch_025.png"))

    def test_make_epoch_output_path_defaults_to_png_suffix(self) -> None:
        output_path = make_epoch_output_path(Path("outputs/circle"), 0, 4)

        self.assertEqual(output_path, Path("outputs/circle_epoch_0.png"))


if __name__ == "__main__":
    unittest.main()
