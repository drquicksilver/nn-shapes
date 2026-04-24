import unittest

import torch

from main import ShapeClassifier, label_box, label_circle


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


if __name__ == "__main__":
    unittest.main()
