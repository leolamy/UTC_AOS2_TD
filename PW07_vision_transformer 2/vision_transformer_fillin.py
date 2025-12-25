# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /home/sylvain/.local/share/jupyter/kernels/python3
# ---

# %% [markdown]
# # Vision transformer
#
# ## Preliminaries
#
# ### Libraries and imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# %% [markdown]
# ### Global variables

# %%
# MNIST images are 28x28
IMAGE_SIZE = 28

# Divide image into (28/7)x(28/7) patches
PATCH_SIZE = 7
NUM_SPLITS = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_SPLITS ** 2

BATCH_SIZE = ...
EMBEDDING_DIM = ...
NUM_HEADS = ...
NUM_CLASSES = ...
NUM_TRANSFORMER_LAYERS = ...
HIDDEN_DIM = ...
EPOCHS = ...
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## The `MNIST` dataset
#
# See [here](https://en.wikipedia.org/wiki/MNIST_database) for details.

# %%
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# %%
# Normalize and reshape
X = X / 255.0
X = X.reshape(-1, 1, 28, 28)  # shape: (n_samples, channels, height, width)

# %%
# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

# %%
# Convert to Pytorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# %%
# Use dataloader to generate minibatches
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1000, shuffle=False)


# %% [markdown]
# ## Patch Embedding Layer
#
# The first module to implement is a module that will transformed a tensor
# of size `BATCH_SIZE` \* 1 \* `IMAGE_SIZE` \* `IMAGE_SIZE` into a tensor
# of size `BATCH_SIZE` \* `NUM_PATCHES` \* `EMBEDDING_DIM`. This can be
# done by using a `nn.Conv2d` module with both the stride and the kernel
# the size of a patch.

# %%
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=7, embedding_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        # Use `nn.Conv2d` to split the image into patches
        self.projection = ...

    def forward(self, x):
        # `x` is `BATCH_SIZE` * 1 * `IMAGE_SIZE` * `IMAGE_SIZE`

        # Project `x` into a tensor of size `BATCH_SIZE` * `EMBEDDING_DIM` *
        # `NUM_SPLITS` * `NUM_SPLITS`
        x = ...

        # Flatten spatial dimensions to have a tensor of size `BATCH_SIZE` *
        # `EMBEDDING_DIM` * `NUM_PATCHES`
        x = ...

        # Put the `NUM_PATCHES` dimension at the second place to have a tensor
        # of size `BATCH_SIZE` * `NUM_PATCHES`` * `EMBEDDING_DIM`
        x = ...

        return x


# %% [markdown]
# ## Transformer encoder

# %%
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super().__init__()
        # Define a `nn.MultiheadAttention` module with `embedding_dim` and
        # `num_heads`. Don't forget to set `batch_first` to `True`
        self.attention = ...

        # Define the position-wise feed-forward network using an `nn.Sequential`
        # module, which consists of a linear layer, a GELU activation function,
        # and another linear layer
        self.mlp = ...

        # Define two layer normalization modules
        self.layernorm1 = ...
        self.layernorm2 = ...

    def forward(self, x):
        # Compute self-attention on `x`
        attn_output, _ = ...

        # Skip-connection and first layer normalization
        x = ...

        # Apply the position-wise feed-forward network
        mlp_output = ...

        # Skip-connection and second layer normalization
        x = ...

        return x


# %% [markdown]
# ## Vision Transformer

# %%
class VisionTransformer(nn.Module):
    def __init__(
            self,
            patch_size,
            embedding_dim,
            num_heads,
            num_classes,
            num_transformer_layers,
            hidden_dim,
    ):
        super().__init__()

        # Define a `PatchEmbedding` module
        self.patch_embedding = ...

        # Use `nn.Parameter` to define an additional token embedding that will
        # be used to predict the class
        self.cls_token = ...

        # Define `position_embedding` and initialize with `nn.init.xavier_uniform_`
        position_embedding = ...
        ...

        # Use `nn.Parameter` to make it learnable
        self.position_embedding = ...

        # Define a sequence of `TransformerEncoder` modules using `nn.Sequential`
        self.encoder_layers = ...

        # Define the classification head as a sequence of a layer normalization
        # followed by a linear transformation mapping `embedding_dim` to
        # `num_classes`
        self.mlp_head = ...

    def forward(self, x):
        # `x` is `BATCH_SIZE` * 1 * `IMAGE_SIZE` * `IMAGE_SIZE`

        # Transform images into embedded patches. It gives a tensor of size
        # `BATCH_SIZE` * `NUM_PATCHES` * `EMBEDDING_DIM`
        x = ...

        # We need to add the embedded classification token at the beginning of
        # each sequence in the minibatch. Use `expand` to duplicate it along the
        # batch size dimension
        batch_size = ...
        cls_tokens = ...

        # Next use `torch.cat` to concatenate `cls_tokens` and `x` to have a
        # tensor of size `BATCH_SIZE` * (NUM_PATCHES + 1) * `EMBEDDING_DIM`
        x = ...

        # Add the positional encoding
        x += ...

        # Apply the stacked transformer modules
        y = ...

        # Select the classification token for each sample in the minibatch.
        # `cls_output` should be of size `BATCH_SIZE` * 1 * `EMBEDDING_DIM`
        cls_output = ...

        # Use `self.mlp_head` to adapt the output size to NUM_CLASSES.
        out = ...

        return out


# %% [markdown]
# ## Initialize model, loss and optimizer

# %%
# Define the `VisionTransformer` model
model = ...

# Use cross-entropy loss and AdamW optimizer with a learning rate of 5e-4
criterion = ...
optimizer = ...


# %% [markdown]
# ## Validation loss calculation

# %%
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


# %% [markdown]
# ## Training with Validation

# %%
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate validation loss and accuracy
    val_loss, val_accuracy = validate_model(model, test_loader, criterion)

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# %% [markdown]
# ## Visualize predictions

# %%
model.eval()
all_images, all_preds, all_labels = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)

        all_images.append(X_batch.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

# Stack all test data
X_all = np.concatenate(all_images)
y_all = np.concatenate(all_labels)
p_all = np.concatenate(all_preds)

# Find wrong predictions
wrong_idx = np.where(p_all != y_all)[0]

fig, axes = plt.subplots(5, 5, figsize=(8, 8))
fig.suptitle("MNIST Wrong Predictions", fontsize=14)

for ax, idx in zip(axes.flat, wrong_idx):
    img = X_all[idx][0]
    true_label = y_all[idx]
    pred_label = p_all[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"T:{true_label} / P:{pred_label}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Further Improvements
#
# 1.  Could you change the attention mechanism to only receive attention
#     from adjacent patches?
