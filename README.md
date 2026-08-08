

# 🚀 Parallel-MetaLearn: Blazing-Fast, VMAP-Powered Functional Meta-Learning for PyTorch

**Stop writing slow `for` loops over your meta-batches. Stop rewriting your PyTorch models into awkward functional syntax.**

MetaLearn is a next-generation, high-performance meta-learning framework built natively on top of PyTorch 2.0+ `torch.func`. Designed for researchers and production engineers, it delivers **massive speedups** by vectorizing the outer-loop task processing while keeping your code clean, modular, and purely object-oriented.

Whether you are doing Few-Shot Classification, Domain Adaptation, Semantic Segmentation, or Regression, MetaLearn adapts to your task—not the other way around.

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

Currently, the library natively supports the most powerful gradient-based meta-learning algorithms out of the box:

* ✅ **MAML** (Model-Agnostic Meta-Learning)
* ✅ **FOMAML** (First Order MAML)
* ✅ **ANIL** (Almost No Inner Loop)
* ✅ **Meta-SGD** (Learnable inner learning rates)
* ✅ **MAML++** (Multi-Step Loss Optimization & Per-step parameters)

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

# 2. Setup Class-Agnostic Encoding & Loss
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

* **Recurrent Network Support:** Native, `vmap`-safe support for `LSTM` and `GRU` layers.
* **Advanced Noise Management:** Robust meta-learning under input perturbations.
* **New Meta-Algorithms:** Integration of cutting-edge algorithms (e.g., ProtoMAML, Reptile).
* **New Inner Optimizers:** Second-order approximation optimizers and adaptive inner-loop schedulers.

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

## 🤝 Contributing & Citation

If you use MetaLearn in your research or production pipelines, we'd love to hear about it! Contributions, issues, and feature requests are always welcome.

> *Fully functional example available in `main.py`.* Just run `python main.py` and watch the `vmap` magic happen!