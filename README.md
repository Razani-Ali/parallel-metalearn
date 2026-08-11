# 🚀 Parallel-MetaLearn: Blazing-Fast, VMAP-Powered Functional Meta-Learning for PyTorch

[![PyPI Version](https://img.shields.io/pypi/v/parallel-metalearn?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/parallel-metalearn/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15IULy7fsm93wtwyXdWapBq2seVluySfC?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Stop writing slow `for` loops over your meta-batches. Stop rewriting your PyTorch models into awkward functional syntax.**

MetaLearn is a next-generation, high-performance meta-learning framework built natively on top of PyTorch 2.0+ `torch.func`. Designed for researchers and production engineers, it delivers **massive speedups** by vectorizing outer-loop task processing while keeping your code clean, modular, and purely object-oriented.

---

## ⚡ Quick Links & Interactive Demo

* 📦 **PyPI Package:** [`pip install parallel-metalearn`](https://pypi.org/project/parallel-metalearn/)
* 🚀 **Interactive Google Colab Notebook:** [Try in Google Colab](https://colab.research.google.com/drive/15IULy7fsm93wtwyXdWapBq2seVluySfC?usp=sharing)

> 💡 **Educational Notebook Notice:**  
> The provided Google Colab notebook is a **demonstration and educational pipeline** designed for fast trial runs on fault diagnosis datasets. The complete core framework and advanced production modules are available in this repository or provided upon request.

---

## 💻 Installation

### Option 1: Install via PyPI (Recommended)
```bash
pip install parallel-metalearn

```

### Option 2: Clone for Local Development & Research

```bash
git clone [https://github.com/your-username/parallel-metalearn.git](https://github.com/your-username/parallel-metalearn.git)
cd parallel-metalearn
pip install -e .

```


---

## 🔥 Why Choose Parallel-MetaLearn? (The Game Changers)

Existing libraries (like `learn2learn` or `higher`) force you into difficult compromises: they either use sequential `for` loops that bottleneck your GPU, or they require you to completely rewrite your model's forward pass to accept explicit parameters (e.g., `torch.functional.conv1d(x, weight=params['w'])`).

**MetaLearn solves all of this:**

* ⚡ **True Parallelism via `vmap`:** We eliminated the task `for` loop. By leveraging PyTorch's `vmap`, MetaLearn processes the entire meta-batch simultaneously. Expect speedups directly proportional to your task batch size (e.g., up to **Q-times faster** where Q is the number of tasks).
* 🧠 **Zero-Friction Model Definitions:** Write your `nn.Module` exactly as you normally would. No need to pass parameter dictionaries into your `forward()` method. We handle the stateless functional calls completely under the hood.
* 🎭 **Dynamic Task Imbalance & Masking:** `vmap` usually crashes if tasks have different batch sizes. We engineered a robust **Masking & Padding engine** under the hood. You can now train on highly imbalanced tasks (`support_shot=(min_shot,max_shot)`) without breaking vectorization!
* 🎯 **Class-Agnostic & Class-Specific Modes:** Seamlessly switch between Class-Agnostic encoding (perfect for Out-Of-Distribution (OOD) generalization to unseen classes) and standard Class-Specific targets.
* 🧩 **Task-Agnostic Architecture:** MetaLearn doesn't care if you are doing Classification, Regression, or Segmentation. Just swap out the Dataset and Loss classes. The core MAML remain 100% untouched.
* ⏱️ **Step-Aware Inner Loop:** Your inner models and optimizers can be fully aware of the current gradient step, allowing for per-step learning rates and independent buffer management (crucial for MAML++).

---

## 🛠️ Supported Algorithms

Currently, the library natively supports a comprehensive suite of gradient-based, metric-based, and first-order meta-learning algorithms out of the box:

* ✅ **MAML** (Model-Agnostic Meta-Learning)
* ✅ **FOMAML** (First-Order MAML)
* ✅ **ANIL** (Almost No Inner Loop)
* ✅ **BOIL** (Body-Only Inner Loop)
* ✅ **Meta-SGD** (Learnable per-layer inner learning rates)
* ✅ **MAML++** (Multi-Step Loss Optimization & per-step learnable parameters)
* ✅ **ProtoMAML (v1 & v2)** (Prototypical MAML featuring First/Second-Order derivatives, body-only updates, multi-step loss accumulation, and per-layer & per-step learnable inner learning rates)
* ✅ **Prototypical Networks** (ProtoNet with customizable & learnable distance metrics)
* ✅ **Reptile** (Fast, first-order weight-delta meta-optimization)
---

## 📦 Core Features at a Glance

* **Customizable Data Pipelines:** Use our highly flexible `MetaTaskDataset` to randomly or deterministically sample N-way K-shot tasks, or easily subclass it for your own custom data logic.
* **Plug-and-Play Optimizers:** Build your own custom Inner-Optimizers effortlessly, and use any standard PyTorch optimizer (Adam, SGD, etc.) for the Outer-Loop.
* **Automated Pipeline:** Say goodbye to boilerplate code. Our `MetaTrain` engine automatically handles the meta-training loop, validation intervals, metric logging, early stopping, and checkpoint saving.

---

## 🚀 Quick Start

The complete pipeline works out of the box. Check out `main.py` for a fully working example on the CWRU Fault Diagnosis dataset. Here is how simple it is to initialize and train:

```python
import torch
from metalearn.model_wrappers import MAML_Model
from metalearn.loss import LabelEncoder, CrossEntropy, CategoricalAccuracy
from metalearn.inner_optimizers import InnerSGD
from metalearn.algorithms import MAML
from metalearn.train import MetaTrain

# 1. Define your standard PyTorch models (No functional rewrites needed!)
backbone = MyCNNBackbone() 
head = MyLinearHead()
model = MAML_Model(backbone=backbone, head=head, drop_rate=0.5)

# 2. Setup Class-Agnostic Encoding & Loss (optional)
label_encoder = LabelEncoder(num_classes=10, max_n_way=3, shuffle=True)
loss_fn = CrossEntropy(metric_fn=CategoricalAccuracy())

# 3. Define Optimizers
inner_optimizer = InnerSGD(initial_fast_weights=model.get_fast_weights(), inner_lr=0.01)
outer_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Initialize Algorithm (MAML, ANIL, MAML++, etc.)
algorithm = MAML(
    model=model,
    optimizer=outer_optimizer,
    inner_optimizer=inner_optimizer,
    support_loss_fn=loss_fn,
    encoder=label_encoder,
)

# 5. Train with Automated Logging & Checkpointing!
trainer = MetaTrain(
    TrainLoader=train_loader, 
    ValLoader=val_loader, 
    algorithm=algorithm
)

history, best_metric, best_loss = trainer.train(
    epochs=1500, 
    check_idx=10, 
    log_checkpoint_path="logs"
)

```

---

## 🗺️ Roadmap (Upcoming Features)

We are constantly pushing the boundaries of what is possible in functional meta-learning. In our upcoming releases, look forward to:

* **Advanced Noise Management:** Robust meta-learning under input perturbations.
* **New Meta-Algorithms:** Integration of cutting-edge algorithms (e.g., Siamese, Matching and relational Networks).

---

## 🛠️ Unmatched Extensibility for Researchers (Developer Guide)

MetaLearn is architected around **strict separation of concerns**. The core MAML execution engine operates purely on standardized output dictionaries (`out_dict`) and target dictionaries (`targets`). This means you can extend MetaLearn to cutting-edge research paradigms **without ever touching the core MAML execution loop or `vmap` logic**:

### 1. 🔀 Multi-Task Learning (MTL)
Need joint classification and auxiliary regression/reconstruction?
* **Data:** Return auxiliary targets alongside labels in `MetaTaskDataset` (e.g., `y_dict = {"labels": y, "reg_targets": reg_y}`).
* **Loss:** Subclass `BaseLoss` to compute composite loss (`cls_loss + lambda * reg_loss`).
* *MAML engine automatically propagates gradients across all tasks!*

### 2. 🌐 Meta-Domain Adaptation (MDA)
Want to align feature distributions across shifting domains?
* **Data:** Pass domain indicators inside your dataset targets (e.g., `y_dict = {"labels": y, "domain_id": d}`).
* **Loss:** Extract features from `out_dict["features"]` and compute domain alignment loss (e.g., MMD, Wasserstein Distance, or Adversarial Loss) inside your custom Loss class.

### 3. 🌐 Federated Meta-Learning (FedMeta)
Want to simulate decentralized client adaptation or privacy-preserving meta-learning?
* **Data & Algorithm:** Keep the same functional `MAML` step, but customize the task assignment logic to simulate client-side local updates before global aggregation.

### 4. 🌐 Federated Learning (FedAvg, FedGrad, FedProx) & FedMeta
Because MetaLearn processes inner-loop updates in a stateless, functional manner, you can effortlessly simulate **Pure Federated Learning algorithms** (e.g., FedAvg, FedGrad) alongside **Federated Meta-Learning (FedMeta)**:
* **Parallel Client Simulation via `vmap`:** Instead of sequentially looping through individual clients, MetaLearn simulates dozens of local client updates *simultaneously* on the GPU using `vmap`.
* **Zero-Overhead Aggregation:** Extract adapted local parameters $\theta_i'$ from each client task, perform global server aggregation (e.g., weighted averaging via `torch.stack(client_weights).mean(dim=0)`), and seamlessly set the new global start state for the next communication round.

---

## 📊 Experimental Setup & Benchmark Results

> ⚠️ **Educational Colab Notice:**  
> The results below are obtained from a fast demonstration run using the provided **Google Colab Notebook** on the CWRU fault diagnosis dataset. It serves as an empirical verification of parallel speed, convergence stability, and meta-generalization capabilities across algorithms under identical runtime constraints.

---

### 1. Dataset & File-Level Stratified Splitting Strategy

To strictly prevent data leakage between meta-training and meta-validation domains, signals are partitioned strictly at the **physical file/session level** rather than random sample-level slicing:

* **Signal Processing:** Raw 2-channel vibration signals segmented into time-series windows of length **2048** with 75% overlap (stride = 512).
* **Domain Partition:** **20% of files** allocated for Meta-Training and **80% of files** reserved for Zero-Shot Meta-Validation (Domain-Shift evaluation).
* **Task Configuration:** 3-Way 5-Shot Support ($K_s=5$) and 15-Shot Query ($K_q=15$) sampled dynamically per episode batch.

---

### 2. Feature Extractor Architecture (Backbone)

A lightweight functional network designed for processing raw vibration sequences:
* **Feature Projection:** Linear layers ($64 \times 4 \to 128 \to 3$) processing chunked signal windows.
* **Normalization:** Custom VMAP-friendly **Step-Aware `BatchNorm`** (`use_per_step_stats=True`) tracking independent running statistics across inner-loop adaptation steps.
* **Embedding Projection:** Global linear mapping producing a **64-dimensional latent representation**.

---

### 3. Experimental Benchmark Comparison (200 Epochs)

All algorithms were trained under identical hardware constraints using a **Meta-Batch Size of 24 tasks** vectorized via `torch.func.vmap`.

| Algorithm | Inner Loop Setup | Best Val Accuracy | Validation Convergence Profile |
| :--- | :--- | :---: | :--- |
| **ProtoMAML v2** | Pure Prototypical Backbone (3 Steps) | **100.00%** | Smooth, rapid convergence & perfect Domain generalization |
| **ProtoMAML v1** | Prototype Head Init + Full SGD (1 Step) | **99.44%** | Highly accurate, steady loss minimization |
| **MAML++** | Learnable LRs + Multi-Step Loss (3 Steps) | **98.89%** | Fast convergence & stable accuracy curve |
| **Prototypical Net** | Non-parametric Distance Alignment | **86.11%** | Underfitting due to lack of inner-loop parameter adaptation |
| **MAML (Vanilla)** | Standard First/Second-Order SGD (3 Steps) | **82.22%** | Noisy convergence with high gradient step variance |
| **Reptile** | First-Order Directional Update (3 Steps) | **70.56%** | Stable train/val alignment but slower adaptation rate |

---

### ⚡ Key Takeaways
1. **Dynamic Prototype Initialization (ProtoMAML v2):** Eliminates gradient noise on classification heads, yielding **100% accuracy** with smooth loss minimization.
2. **Step-Aware BatchNorm & Multi-Step Loss (MAML++):** Stabilizes standard gradient-based MAML, boosting accuracy from **82.22% to 98.89%**.
3. **Blazing Execution Speed:** Processing 200 epochs of full second-order MAML++ optimization across 24 parallelized tasks completes in **~23.65 seconds** on a single GPU thanks to `vmap` vectorization.

---
## 🤝 Contributing & Citation

If you use MetaLearn in your research or production pipelines, we'd love to hear about it! Contributions, issues, and feature requests are always welcome.

> *Fully functional example available in `main.py`.* Just run `python main.py` and watch the `vmap` magic happen!