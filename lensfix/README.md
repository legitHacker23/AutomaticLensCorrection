# LensFix — Automatic Lens Correction

Hybrid geometric warp model that corrects lens distortion using paired training data.
Optimised for edge/line/gradient metrics (85 % of competition score).

## Repo structure

```
lensfix/
├── configs/default.yaml   # all hyper-parameters
├── src/
│   ├── dataset.py         # paired + test datasets
│   ├── model.py           # HybridWarpNet
│   ├── losses.py          # composite loss
│   ├── train.py           # training loop
│   ├── infer.py           # inference at full resolution
│   └── utils.py           # shared helpers
└── scripts/
    └── make_zip.py        # package output for submission
```

## Quick start (Colab / A100)

Run all commands from the `lensfix/` directory.

```bash
# 0. Clone / upload repo, then cd into it
cd lensfix

# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data (see "Data layout" below)
#    e.g. upload or symlink into data/train and data/test

# 3. Stage-1 training (256 px, ~10 epochs)
python -m src.train --config configs/default.yaml --stage train

# 4. Stage-2 finetune (384 px, ~5 epochs, resumes from best stage-1 ckpt)
python -m src.train --config configs/default.yaml \
    --stage finetune --resume checkpoints/best.pt

# — OR run both stages back-to-back —
python -m src.train --config configs/default.yaml --stage both

# 5. Inference on test set (full resolution by default)
python -m src.infer --config configs/default.yaml

# 6. (optional) Fixed-res mode if GPU memory is tight
python -m src.infer --config configs/default.yaml \
    --no-full-res --infer-size 384

# 7. Create submission zip
python scripts/make_zip.py --input output --output submission.zip
```

### Override paths at runtime

```bash
python -m src.infer --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --output output
```

## Data layout

Place data so that one of these structures exists:

**Layout A** (per-ID folders):
```
data/train/<id>/original.jpg
data/train/<id>/generated.jpg
```

**Layout B** (split folders):
```
data/train/original/<id>.jpg
data/train/generated/<id>.jpg
```

Test images:
```
data/test/<image_id>.jpg
```

The dataset loader auto-detects which layout is present.

## Inference modes

| Mode | Flag | Description |
|---|---|---|
| Full-res (default) | *(none)* | Runs model at each image's native H×W. Best quality; needs sufficient VRAM. |
| Fixed-res | `--no-full-res --infer-size 384` | Predicts warp at 384×384, upsamples grid to native H×W, applies to full-res image. Lower VRAM, slight quality trade-off. |

Both modes produce output images at the **exact original resolution** with **identical filenames**.
