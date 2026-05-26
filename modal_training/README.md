# Modal Training

YOLOv4-tiny training pipeline running on Modal (H100 GPU for training and evaluation, CPU for build and upload steps), uploading results to HuggingFace Hub.

## Setup (one-time)

### 1. Install Modal CLI

```bash
pip install modal
modal setup
```

### 2. Create Secrets on Modal Dashboard

Go to https://modal.com/secrets and create:

**Secret name: `roboflow`**
- Key: `ROBOFLOW_API_KEY`
- Value: API key from https://app.roboflow.com (Settings -> Roboflow API)

**Secret name: `huggingface`**
- Key: `HF_TOKEN`
- Value: Token from https://huggingface.co/settings/tokens (Write permission)

### 3. Create a HuggingFace model repository

- Go to https://huggingface.co/new
- Repository name: `yolov4-tiny-hpo-ffb-maturity`
- Visibility: Public
- Full name: `dutaav/yolov4-tiny-hpo-ffb-maturity`

Edit `app.py` and set:

```python
HF_REPO_DEFAULT = "dutaav/yolov4-tiny-hpo-ffb-maturity"
```

## Run

### Full pipeline

```bash
cd modal_training
modal run app.py
```

Actual times from the H100 run:
- Build darknet: ~5 minutes (one-time, cached)
- Dataset download: ~3 minutes (one-time, cached)
- Model 1 + Model 2 (parallel): ~61 min (model1=60.6 min, model2=40.7 min)
- GA (5 generations x 10 parallel fitness evals, 3000 iter each): variable
- Model 3 + Model 4 (parallel): ~57 min (model3=56.6 min, model4=30.3 min)
- Evaluation + upload: ~5 minutes

Total wall clock: ~2-3 hours. Estimated cost: ~$8-12.

### Custom runs

Skip GA and use a predefined learning rate:

```bash
modal run app.py --skip-ga --ga-lr-override 0.007465
```

Custom HF repository:

```bash
modal run app.py --hf-repo myusername/palm-yolov4
```

Longer GA fitness iterations:

```bash
modal run app.py --fitness-iters 3000 --n-generations 5 --n-population 10
```

### Individual steps

```bash
modal run app.py::build_darknet
modal run app.py::prepare_dataset
modal run app.py::upload_to_hf --repo-id dutaav/yolov4-tiny-hpo-ffb-maturity
```

### Force rebuild

```bash
modal run app.py::build_darknet --force
```

## Output

After completion, all artifacts are uploaded to HuggingFace under `runs/{run_tag}/`:

```
runs/h100-hankai/
├── weights/
│   ├── model1_baseline_best.weights
│   ├── model2_es_best.weights
│   ├── model3_ga_best.weights
│   └── model4_es_ga_best.weights
├── configs/
│   ├── model1_baseline.cfg
│   ├── model2_es.cfg
│   ├── model3_ga.cfg
│   └── model4_es_ga.cfg
├── logs/
│   ├── model1_baseline.log
│   ├── ...
│   └── per-split eval outputs
└── artifacts/
    ├── metrics.json
    ├── training_info.json
    ├── ga_history.json
    └── dataset_info.json
```

Continue to `colab_inference/` for visualization in Google Colab.
