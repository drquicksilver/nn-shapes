# nn-shapes

Train a small neural network to classify whether two-dimensional points are inside a simple shape, then plot the learned decision boundary.

## Run locally

```bash
uv run python main.py --shape circle --seed 1001 --output outputs/circle.png
```

Useful parameters include `--shape`, `--epochs`, `--hidden-size`, `--hidden-layers`, `--num-samples`, `--batch-size`, `--learning-rate`, `--seed`, `--output`, and `--plot-every-epochs`.

To save training snapshots, pass a positive epoch interval:

```bash
uv run python main.py --shape circle --seed 1001 --output outputs/circle.png --plot-every-epochs 10
```

This writes the final plot to `outputs/circle.png`, plus snapshots such as `outputs/circle_epoch_0000.png` and `outputs/circle_epoch_0010.png`. Epoch 0 is saved before training so the random initialisation is visible. The filename padding is based on the total epochs for the run.

## Generated image gallery

The repository includes a static gallery pipeline for article-figure iteration:

- `gallery/experiments.json` defines the parameter sets to render.
- `scripts/build_gallery.py` trains each configured experiment, writes PNGs into `site/images/`, writes `site/index.html`, and writes `site/manifest.json` with reproducibility metadata.
- `.github/workflows/gallery.yml` runs the tests with `python -m unittest discover -s tests`, builds the gallery, uploads the generated `site/` directory as a GitHub Pages artifact, and deploys it to GitHub Pages.

Build the gallery locally with:

```bash
uv run python scripts/build_gallery.py --config gallery/experiments.json --site-dir site
```

The generated `site/` contents are ignored by Git except for `site/.gitkeep`, because GitHub Actions regenerates the gallery before publishing it.

## GitHub Pages setup

The workflow is ready to deploy through GitHub Actions once Pages is enabled for the repository:

1. In GitHub, open the repository settings.
2. Go to **Settings → Pages**.
3. Set **Build and deployment → Source** to **GitHub Actions**.
4. Save the Pages settings.
5. Push to `main`, or run **Generate image gallery** manually from the repository's **Actions** tab.

The workflow needs the repository-level default Actions permissions to allow GitHub Pages deployment. If your account or organization has restricted them, go to **Settings → Actions → General → Workflow permissions** and ensure workflows can request the `pages: write` and `id-token: write` permissions declared in the workflow.
