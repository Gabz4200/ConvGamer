# Notebooks

## Syncing `.py` and `.ipynb`

- Edit `notebooks/01_quickstart.py` (preferred — better diffs in Git).
- Sync to `.ipynb` before running locally or uploading to Kaggle:

```bash
uv run jupytext --sync notebooks/
```

- Or generate `.ipynb` one-shot:

```bash
uv run jupytext --to ipynb notebooks/01_quickstart.py
```

## Kaggle upload

1. Upload `01_quickstart.ipynb` to Kaggle.
2. In the Kaggle notebook, install dependencies inline:

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install omegaconf hydra-core pytorch-lightning taichi transformers
```

3. Then add the repo source and import:

```python
import sys

sys.path.append("/kaggle/input/<dataset-name>/src")
from convgamer.models.registry import get_model
```