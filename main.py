# !pip install cwru-plus


import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import cwru
from metalearn.dataset import FewShotSampler, MetaTaskDataset

(X, Y, file_ids), metadata = cwru.load(
    npz_path="CWRU12Ingested.npz", 
    window_size=2048, 
    step_size=512
)

(tr_d, val_d, te_d), _ = cwru.stratified_file_split(
    X=X,
    y=Y, 
    file_ids=file_ids, 
    train_ratio=0.2, 
    val_ratio=0.8, 
    random_seed=42
)

X_train_raw, Y_train_raw = tr_d
X_val_raw, Y_val_raw = val_d

X_tr_supp, X_tr_query, Y_tr_supp, Y_tr_query = train_test_split(
    X_train_raw, Y_train_raw,
    test_size=0.5,
    stratify=Y_train_raw,
    random_state=42
)

X_val_supp, X_val_query, Y_val_supp, Y_val_query = train_test_split(
    X_val_raw, Y_val_raw,
    test_size=0.5,
    stratify=Y_val_raw,
    random_state=42
)


unique_classes = sorted(list(set(Y_train_raw)))
numeric_to_string = {idx: str(cls) for idx, cls in enumerate(unique_classes)}
max_classes = len(unique_classes)

train_support_sampler = FewShotSampler(
    X_base=X_tr_supp, 
    Y_base=Y_tr_supp, 
    numeric_to_string=numeric_to_string
)

train_query_sampler = FewShotSampler(
    X_base=X_tr_query, 
    Y_base=Y_tr_query, 
    numeric_to_string=numeric_to_string
)

val_support_sampler = FewShotSampler(
    X_base=X_val_supp, 
    Y_base=Y_val_supp, 
    numeric_to_string=numeric_to_string
)

val_query_sampler = FewShotSampler(
    X_base=X_val_query, 
    Y_base=Y_val_query, 
    numeric_to_string=numeric_to_string
)

train_dataset = MetaTaskDataset(
    support_sampler=train_support_sampler,
    query_sampler=train_query_sampler,
    max_classes=max_classes,
    way=3,
    support_shot=5,
    query_shot=15,
    imbalanced_shot=False,
)

val_dataset = MetaTaskDataset(
    support_sampler=val_support_sampler,
    query_sampler=val_query_sampler,
    max_classes=max_classes,
    way=3,
    support_shot=5,
    query_shot=15,
    imbalanced_shot=False,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=24,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

x_s, y_s, x_q, y_q = next(iter(train_loader))

print("✅ DataLoaders Built Successfully!")
print(f"X_support Shape : {x_s.shape}")  # Shape: (tasks_per_batch, way * s_shot, ...)
print(f"Y_support Shape : {y_s["labels"].shape}")  # Shape: (tasks_per_batch, way * s_shot)
print(f"X_query Shape   : {x_q.shape}")  # Shape: (tasks_per_batch, way * q_shot, ...)
print(f"Y_query Shape   : {y_q["labels"].shape}")  # Shape: (tasks_per_batch, way * q_shot)


import torch
import torch.nn as nn
from typing import Dict
from metalearn.dataset import RobustScaler
from metalearn.model_wrappers import MAML_Model


class Backbone(nn.Module):
    """
    Lightweight Deep Network for 2-channel 2048-length sensor signals.
    
    Reshapes input (Batch, 2, 2048) -> (Batch, 64, 64), applies channel-wise 
    dimension reduction (64 -> 16 -> 3), flattens features to 192 dimensions,
    and classifies via a global linear head.
    """

    def __init__(self):
        super().__init__()

        # 1. Feature Extraction Layers (applied to chunks of 64 samples)
        self.feature_layer1 = nn.Linear(64, 128)
        self.relu1 = nn.ReLU()

        self.feature_layer2 = nn.Linear(128, 3)
        self.relu2 = nn.ReLU()

        # 2. Global Classification Head (192 input features -> num_classes)
        # 64 chunks * 3 features per chunk = 192 total features
        self.global_feature = nn.Linear(64 * 3, 16)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 2, 2048) or (Batch, 4096)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'logits' and 'features'.
        """
        # 1. Flatten channels and length into a single 4096 axis regardless of batch rank
        # Shape becomes: (..., 4096)
        x_flat = x.flatten(start_dim=-2) if x.dim() >= 2 else x

        # 2. Unflatten the last dimension (4096) into two chunk axes (64, 64)
        # Shape becomes: (..., 64, 64) -- Fully vmap safe!
        x_reshaped = x_flat.unflatten(dim=-1, sizes=(64, 64))

        # 2. Sequential Linear Transformation over the last dimension (64)
        h1 = self.relu1(self.feature_layer1(x_reshaped))  # Shape: (Batch, 64, 16)
        h2 = self.relu2(self.feature_layer2(h1))          # Shape: (Batch, 64, 3)

        # 3. Concatenate/Flatten all chunk outputs -> (Batch, 192)
        features = h2.flatten(start_dim=-2)

        # 4. Global Linear Classification -> (Batch, num_classes)
        g_feat = self.global_feature(features)

        return g_feat


class Head(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(64 * 16, 3)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        logit = self.layer(x)

        return logit

scaler = RobustScaler()
scaler.fit(torch.from_numpy(X_tr_supp))

model = MAML_Model(backbone=Backbone(), head=Head(), scaler=scaler, drop_rate=0.5)


from metalearn.loss import LabelEncoder, CrossEntropy, CategoricalAccuracy

# 1. Instantiate Random LabelEncoder with max_n_way = 3
label_encoder = LabelEncoder(
    num_classes=max_classes,
    max_n_way=3,
    shuffle=True
)

# 2. Instantiate Categorical Cross Entropy Loss with LabelEncoder
loss_fn = CrossEntropy(metric_fn=CategoricalAccuracy())


from metalearn.inner_optimizers import InnerSGD
from metalearn.train import MetaTrain
from metalearn.algorithms import MAML


inner_optimizer = InnerSGD(initial_fast_weights=model.get_fast_weights(),
                           inner_lr=0.01,
                           learn_lr=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

algorithm = MAML(model=model,
                 optimizer=optimizer,
                 inner_optimizer=inner_optimizer,
                 support_loss_fn=loss_fn,
                 chunk_size=train_loader.batch_size,
                 encoder=label_encoder,
                 )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

trainer = MetaTrain(TrainLoader=train_loader,
                    ValLoader=val_loader,
                    scheduler=scheduler,
                    algorithm=algorithm,
                    )


history, best_val_metric, best_val_loss = trainer.train(
    epochs=1500,
    check_idx=10,
    log_checkpoint_path="logs",
    replace_check_point=True,
    
)


import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional


def plot_meta_history(
    history: Dict[str, Any], 
    best_val_metric: Optional[float] = None, 
    best_val_loss: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Plots Meta-Training and Meta-Validation Loss, Accuracy/Metric, and Learning Rate curves.

    Args:
        history (Dict[str, Any]): History dictionary returned by MetaTrain.train().
        best_val_metric (Optional[float]): Best validation metric achieved during training.
        best_val_loss (Optional[float]): Best validation loss achieved during training.
        save_path (Optional[str]): Path to save the output plot figure.
    """
    # Extract training metrics
    train_loss = history.get('train_loss', [])
    train_metric = history.get('train_metric', [])
    val_loss = history.get('val_loss', [])
    val_metric = history.get('val_metric', [])
    learning_rate = history.get('learning_rate', [])

    total_epochs = len(train_loss)
    train_epochs = np.arange(1, total_epochs + 1)

    # Compute exact validation X-axis indices dynamically
    if len(val_loss) > 0:
        val_epochs = np.linspace(1, total_epochs, len(val_loss), dtype=int)
    else:
        val_epochs = []

    # Initialize 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --------------------------------------------------------------------------
    # Panel 1: Loss Curve (Train vs Validation)
    # --------------------------------------------------------------------------
    axes[0].plot(train_epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
    if len(val_loss) > 0:
        axes[0].plot(val_epochs, val_loss, label='Val Loss', color='#ff7f0e', linestyle='--', marker='o', markersize=4)
        if best_val_loss is not None:
            axes[0].axhline(y=best_val_loss, color='#d62728', linestyle=':', label=f'Best Val Loss: {best_val_loss:.4f}')

    axes[0].set_title('Meta-Loss Progression', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].legend(loc='upper right', frameon=True)

    # --------------------------------------------------------------------------
    # Panel 2: Metric / Accuracy Curve (Train vs Validation)
    # --------------------------------------------------------------------------
    axes[1].plot(train_epochs, train_metric, label='Train Metric', color='#2ca02c', linewidth=2)
    if len(val_metric) > 0:
        axes[1].plot(val_epochs, val_metric, label='Val Metric', color='#d62728', linestyle='--', marker='s', markersize=4)
        if best_val_metric is not None:
            axes[1].axhline(y=best_val_metric, color='#9467bd', linestyle=':', label=f'Best Val Acc: {best_val_metric*100:.2f}%')

    axes[1].set_title('Meta-Accuracy / Metric Progression', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs', fontsize=10)
    axes[1].set_ylabel('Metric Value', fontsize=10)
    axes[1].legend(loc='lower right', frameon=True)

    # --------------------------------------------------------------------------
    # Panel 3: Outer Learning Rate Schedule
    # --------------------------------------------------------------------------
    axes[2].plot(train_epochs, learning_rate, label='Outer LR', color='#9467bd', linewidth=2)
    axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epochs', fontsize=10)
    axes[2].set_ylabel('Learning Rate', fontsize=10)
    axes[2].set_yscale('log')  # Logarithmic scale for better LR visibility
    axes[2].legend(loc='upper right', frameon=True)

    # Format layout and display total training time if available
    total_time = history.get('total_elapsed_time', 0.0)
    plt.suptitle(f"Meta-Training Summary (Total Pure Training Time: {total_time:.2f} s)", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"📊 Plot saved successfully to '{save_path}'")

    plt.show()


plot_meta_history(
    history=history,
    best_val_metric=best_val_metric,
    best_val_loss=best_val_loss,
)