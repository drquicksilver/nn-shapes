import argparse
import html
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import (
    SHAPES,
    ShapeClassifier,
    make_dataset,
    make_epoch_output_path,
    plot_decision_boundary,
    set_seed,
    train_model,
)


GALLERY_SNAPSHOT_INTERVAL_EPOCHS = 50


@dataclass(frozen=True)
class Experiment:
    name: str
    title: str
    description: str
    shape: str
    epochs: int
    hidden_size: int
    hidden_layers: int
    num_samples: int
    batch_size: int
    learning_rate: float
    seed: int

    @property
    def image_filename(self) -> str:
        return f"{self.name}.png"

    @property
    def image_path(self) -> str:
        return f"images/{self.image_filename}"

    def command(self, output_path: Path) -> str:
        return " ".join(
            [
                "python main.py",
                f"--shape {self.shape}",
                f"--epochs {self.epochs}",
                f"--hidden-size {self.hidden_size}",
                f"--hidden-layers {self.hidden_layers}",
                f"--num-samples {self.num_samples}",
                f"--batch-size {self.batch_size}",
                f"--learning-rate {self.learning_rate:g}",
                f"--seed {self.seed}",
                f"--output {output_path.as_posix()}",
                f"--plot-every-epochs {GALLERY_SNAPSHOT_INTERVAL_EPOCHS}",
            ]
        )


@dataclass(frozen=True)
class GalleryFrame:
    epoch: int
    image_path: str


@dataclass(frozen=True)
class GalleryItem:
    experiment: Experiment
    final_loss: float
    final_accuracy: float
    output_path: Path
    frames: List[GalleryFrame]

    def metadata(self) -> Dict[str, Any]:
        data = asdict(self.experiment)
        data.update(
            {
                "image_path": self.experiment.image_path,
                "frames": [asdict(frame) for frame in self.frames],
                "final_loss": self.final_loss,
                "final_accuracy": self.final_accuracy,
                "command": self.experiment.command(self.output_path),
            }
        )
        return data


def parse_experiment(raw: Dict[str, Any]) -> Experiment:
    experiment = Experiment(**raw)
    if not experiment.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"experiment name must be filename-safe: {experiment.name!r}")
    if experiment.shape not in SHAPES:
        valid_shapes = ", ".join(sorted(SHAPES))
        raise ValueError(
            f"unknown shape {experiment.shape!r} for {experiment.name!r}; "
            f"expected one of: {valid_shapes}"
        )
    if experiment.epochs <= 0:
        raise ValueError(f"epochs must be positive for {experiment.name!r}")
    if experiment.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive for {experiment.name!r}")
    if experiment.hidden_layers < 0:
        raise ValueError(f"hidden_layers cannot be negative for {experiment.name!r}")
    if experiment.num_samples <= 0:
        raise ValueError(f"num_samples must be positive for {experiment.name!r}")
    if experiment.batch_size <= 0:
        raise ValueError(f"batch_size must be positive for {experiment.name!r}")
    if experiment.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive for {experiment.name!r}")
    return experiment


def load_experiments(config_path: Path) -> List[Experiment]:
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_experiments = json.load(config_file)

    if not isinstance(raw_experiments, list):
        raise ValueError("gallery config must contain a list of experiments")

    experiments = [parse_experiment(raw) for raw in raw_experiments]
    names = [experiment.name for experiment in experiments]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate experiment names: {', '.join(duplicate_names)}")
    return experiments


def run_experiment(experiment: Experiment, site_dir: Path) -> GalleryItem:
    output_path = site_dir / experiment.image_path
    frames: List[GalleryFrame] = []
    set_seed(experiment.seed)
    dataset = make_dataset(experiment.shape, experiment.num_samples)
    model = ShapeClassifier(
        hidden_size=experiment.hidden_size,
        hidden_layers=experiment.hidden_layers,
    )

    def save_epoch_frame(epoch: int, epoch_result: Any) -> None:
        frame_output_path = make_epoch_output_path(output_path, epoch, experiment.epochs)
        plot_decision_boundary(
            model=model,
            shape=experiment.shape,
            seed=experiment.seed,
            result=epoch_result,
            output_path=frame_output_path,
            show=False,
            training_epoch=epoch,
        )
        frames.append(
            GalleryFrame(
                epoch=epoch,
                image_path=frame_output_path.relative_to(site_dir).as_posix(),
            )
        )

    result = train_model(
        model=model,
        dataset=dataset,
        epochs=experiment.epochs,
        batch_size=experiment.batch_size,
        learning_rate=experiment.learning_rate,
        snapshot_interval_epochs=GALLERY_SNAPSHOT_INTERVAL_EPOCHS,
        snapshot_callback=save_epoch_frame,
    )
    plot_decision_boundary(
        model=model,
        shape=experiment.shape,
        seed=experiment.seed,
        result=result,
        output_path=output_path,
        show=False,
    )
    if not frames or frames[-1].epoch != experiment.epochs:
        frames.append(
            GalleryFrame(
                epoch=experiment.epochs,
                image_path=experiment.image_path,
            )
        )
    return GalleryItem(
        experiment=experiment,
        final_loss=result.final_loss,
        final_accuracy=result.final_accuracy,
        output_path=output_path,
        frames=frames,
    )


def render_card(item: GalleryItem) -> str:
    experiment = item.experiment
    initial_frame = item.frames[-1] if item.frames else GalleryFrame(
        epoch=experiment.epochs,
        image_path=experiment.image_path,
    )
    image_path = html.escape(initial_frame.image_path, quote=True)
    frame_sources = html.escape(
        json.dumps([frame.image_path for frame in item.frames]),
        quote=True,
    )
    frame_labels = html.escape(
        json.dumps([f"Epoch {frame.epoch}" for frame in item.frames]),
        quote=True,
    )
    frame_max = max(len(item.frames) - 1, 0)
    frame_value = frame_max
    frame_label = html.escape(f"Epoch {initial_frame.epoch}")
    title = html.escape(experiment.title)
    description = html.escape(experiment.description)
    command = html.escape(experiment.command(item.output_path))
    return f"""
      <article class="card" data-scrubber-card data-frame-srcs="{frame_sources}" data-frame-labels="{frame_labels}">
        <a class="figure-link" href="{image_path}"><img class="figure-image" loading="lazy" src="{image_path}" alt="{title} decision boundary"></a>
        <div class="card-body">
          <h2>{title}</h2>
          <p>{description}</p>
          <label class="scrubber">
            <span>Training epoch</span>
            <input class="frame-scrubber" type="range" min="0" max="{frame_max}" value="{frame_value}" step="1">
            <output class="frame-label">{frame_label}</output>
          </label>
          <dl>
            <div><dt>Shape</dt><dd>{html.escape(experiment.shape)}</dd></div>
            <div><dt>Seed</dt><dd>{experiment.seed}</dd></div>
            <div><dt>Epochs</dt><dd>{experiment.epochs}</dd></div>
            <div><dt>Hidden size</dt><dd>{experiment.hidden_size}</dd></div>
            <div><dt>Hidden layers</dt><dd>{experiment.hidden_layers}</dd></div>
            <div><dt>Samples</dt><dd>{experiment.num_samples}</dd></div>
            <div><dt>Learning rate</dt><dd>{experiment.learning_rate:g}</dd></div>
            <div><dt>Final loss</dt><dd>{item.final_loss:.4f}</dd></div>
            <div><dt>Final accuracy</dt><dd>{item.final_accuracy:.3f}</dd></div>
          </dl>
          <details>
            <summary>Reproduce locally</summary>
            <pre><code>{command}</code></pre>
          </details>
        </div>
      </article>"""


def render_index(items: Iterable[GalleryItem]) -> str:
    cards = "\n".join(render_card(item) for item in items)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nn-shapes gallery</title>
  <style>
    :root {{
      color-scheme: light dark;
      --background: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --code: #f1f5f9;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --background: #020617;
        --card: #0f172a;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --border: #1e293b;
        --code: #111827;
      }}
    }}
    body {{
      margin: 0;
      background: var(--background);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem 1rem 4rem;
    }}
    header {{
      margin-bottom: 2rem;
    }}
    h1 {{
      margin-bottom: 0.5rem;
      font-size: clamp(2rem, 5vw, 3.5rem);
      line-height: 1;
    }}
    .intro {{
      max-width: 760px;
      color: var(--muted);
      font-size: 1.1rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.5rem;
    }}
    .card {{
      overflow: hidden;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 1rem;
      box-shadow: 0 8px 24px rgb(15 23 42 / 8%);
    }}
    .card img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }}
    .card-body {{
      padding: 1rem;
    }}
    .card h2 {{
      margin: 0 0 0.25rem;
    }}
    .card p {{
      margin: 0 0 1rem;
      color: var(--muted);
    }}
    .scrubber {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.35rem 0.75rem;
      align-items: center;
      margin: 0 0 1rem;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .scrubber input {{
      grid-column: 1 / -1;
      width: 100%;
    }}
    .scrubber output {{
      color: var(--text);
      font-weight: 650;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.5rem 1rem;
      margin: 0 0 1rem;
    }}
    dl div {{
      min-width: 0;
    }}
    dt {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    dd {{
      margin: 0;
      font-weight: 650;
    }}
    details {{
      border-top: 1px solid var(--border);
      padding-top: 0.75rem;
    }}
    summary {{
      cursor: pointer;
      color: var(--muted);
    }}
    pre {{
      overflow-x: auto;
      margin-bottom: 0;
      padding: 0.75rem;
      background: var(--code);
      border-radius: 0.5rem;
      font-size: 0.82rem;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>nn-shapes generated image gallery</h1>
      <p class="intro">Decision-boundary images generated by GitHub Actions from the checked-in experiment configuration. Use this page to compare article figures during iteration.</p>
    </header>
    <section class="grid" aria-label="Generated decision-boundary figures">
{cards}
    </section>
  </main>
  <script>
    for (const card of document.querySelectorAll("[data-scrubber-card]")) {{
      const sources = JSON.parse(card.dataset.frameSrcs || "[]");
      const labels = JSON.parse(card.dataset.frameLabels || "[]");
      const image = card.querySelector(".figure-image");
      const link = card.querySelector(".figure-link");
      const scrubber = card.querySelector(".frame-scrubber");
      const output = card.querySelector(".frame-label");

      function showFrame(index) {{
        const source = sources[index];
        if (!source) {{
          return;
        }}
        image.src = source;
        link.href = source;
        output.textContent = labels[index] || "";
      }}

      scrubber.addEventListener("input", () => showFrame(Number(scrubber.value)));
      showFrame(Number(scrubber.value));
    }}
  </script>
</body>
</html>
"""


def write_manifest(items: Iterable[GalleryItem], site_dir: Path) -> None:
    manifest = [item.metadata() for item in items]
    with (site_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


def build_gallery(config_path: Path, site_dir: Path) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "images").mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(config_path)
    items = [run_experiment(experiment, site_dir) for experiment in experiments]
    (site_dir / "index.html").write_text(render_index(items), encoding="utf-8")
    write_manifest(items, site_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the static nn-shapes GitHub Pages gallery."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("gallery/experiments.json"),
        help="Path to the gallery experiment configuration JSON.",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path("site"),
        help="Directory where the generated static site should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_gallery(args.config, args.site_dir)


if __name__ == "__main__":
    main()
