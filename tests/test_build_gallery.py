import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_gallery import (
    Experiment,
    GalleryFrame,
    GalleryItem,
    load_experiments,
    render_index,
    write_manifest,
)


class GalleryConfigTests(unittest.TestCase):
    def write_config(self, directory: Path, experiments: object) -> Path:
        config_path = directory / "experiments.json"
        config_path.write_text(json.dumps(experiments), encoding="utf-8")
        return config_path

    def valid_experiment(self, name: str = "circle_test") -> dict:
        return {
            "name": name,
            "title": "Circle test",
            "description": "A deterministic test experiment.",
            "shape": "circle",
            "epochs": 1,
            "hidden_size": 4,
            "hidden_layers": 1,
            "num_samples": 8,
            "batch_size": 4,
            "learning_rate": 0.001,
            "seed": 123,
        }

    def test_load_experiments_parses_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(Path(temp_dir), [self.valid_experiment()])

            experiments = load_experiments(config_path)

        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].name, "circle_test")
        self.assertEqual(experiments[0].image_path, "images/circle_test.png")

    def test_load_experiments_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                Path(temp_dir),
                [self.valid_experiment("duplicate"), self.valid_experiment("duplicate")],
            )

            with self.assertRaisesRegex(ValueError, "duplicate experiment names"):
                load_experiments(config_path)

    def test_load_experiments_rejects_unknown_shape(self) -> None:
        experiment = self.valid_experiment()
        experiment["shape"] = "triangle"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(Path(temp_dir), [experiment])

            with self.assertRaisesRegex(ValueError, "unknown shape"):
                load_experiments(config_path)


class GalleryRenderingTests(unittest.TestCase):
    def make_item(self) -> GalleryItem:
        experiment = Experiment(
            name="circle_html",
            title="Circle <baseline>",
            description="Check escaping & metadata.",
            shape="circle",
            epochs=2,
            hidden_size=4,
            hidden_layers=1,
            num_samples=16,
            batch_size=4,
            learning_rate=0.01,
            seed=42,
        )
        return GalleryItem(
            experiment=experiment,
            final_loss=0.12345,
            final_accuracy=0.98765,
            output_path=Path("site/images/circle_html.png"),
            frames=[
                GalleryFrame(epoch=0, image_path="images/circle_html_epoch_0.png"),
                GalleryFrame(epoch=2, image_path="images/circle_html_epoch_2.png"),
            ],
        )

    def test_render_index_includes_escaped_experiment_metadata(self) -> None:
        html = render_index([self.make_item()])

        self.assertIn("Circle &lt;baseline&gt;", html)
        self.assertIn("Check escaping &amp; metadata.", html)
        self.assertIn("Final accuracy", html)
        self.assertIn("0.988", html)
        self.assertIn("images/circle_html.png", html)
        self.assertIn('data-scrubber-card', html)
        self.assertIn('value="1"', html)
        self.assertIn("images/circle_html_epoch_2.png", html)
        self.assertIn("Epoch 2", html)

    def test_write_manifest_records_reproducibility_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_dir = Path(temp_dir)
            write_manifest([self.make_item()], site_dir)

            manifest = json.loads((site_dir / "manifest.json").read_text())

        self.assertEqual(manifest[0]["name"], "circle_html")
        self.assertEqual(manifest[0]["image_path"], "images/circle_html.png")
        self.assertEqual(
            manifest[0]["frames"],
            [
                {"epoch": 0, "image_path": "images/circle_html_epoch_0.png"},
                {"epoch": 2, "image_path": "images/circle_html_epoch_2.png"},
            ],
        )
        self.assertEqual(manifest[0]["final_loss"], 0.12345)
        self.assertIn("--seed 42", manifest[0]["command"])
        self.assertIn("--output site/images/circle_html.png", manifest[0]["command"])
        self.assertIn("--plot-every-epochs 50", manifest[0]["command"])


if __name__ == "__main__":
    unittest.main()
