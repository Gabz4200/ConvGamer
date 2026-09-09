# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # InceptionNeXt Quickstart
#
# Loads the package, builds a model, runs one forward pass.

# %%
import torch

from convgamer.models.registry import get_model

# %%
model = get_model("encoder", input_dim=3, hidden_dim=32, num_layers=1, num_classes=10)
x = torch.randn(2, 3, 32, 32)
y = model(x)
print(y.shape)
