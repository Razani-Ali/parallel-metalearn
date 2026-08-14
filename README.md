# Parallel-MetaLearn: A Functional, Vectorized Meta-Learning Framework in PyTorch

[![PyPI Version](https://img.shields.io/pypi/v/parallel-metalearn?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/parallel-metalearn/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15IULy7fsm93wtwyXdWapBq2seVluySfC?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**Parallel-MetaLearn** is a modular PyTorch framework designed for gradient-based and metric-based meta-learning research. By leveraging the functional transformation primitives of `torch.func` (specifically `vmap`, `grad`, and `functional_call`), the framework parallelizes task-level inner adaptation loops across the meta-batch dimension.

Standard meta-learning implementations typically iterate sequentially over tasks within a meta-batch using explicit Python loops, causing suboptimal GPU utilization, or require rewriting model architectures into non-standard functional forms. `Parallel-MetaLearn` preserves standard object-oriented PyTorch `nn.Module` definitions while vectorizing inner-loop optimization paths via stateless execution.

---

## Key Methodological Features

* **Task-Level Vectorization (`torch.func.vmap`):** Inner adaptation steps across independent tasks within an episode are evaluated in parallel, significantly reducing dispatch overhead.
* **Standard `nn.Module` Compatibility:** Model definitions use standard PyTorch layers without manual functional parameter passing in `forward()`.
* **Stateful Buffer Tracking:** Supports per-step running statistics (e.g., in `BatchNorm`) and prototype tracking across both first-order and second-order derivative passes.
* **Support for Task Imbalance & Dynamic Masking:** Includes a masking and padding engine allowing variable support/query shot allocations per episode without violating vectorization constraints.
* **Ghost Graph Suppression:** Incorporates early weight detachment and explicit graph truncation in first-order modes (e.g., FOMAML, Reptile) and evaluation routines to prevent memory leakage.
* **Modular Extensibility:** Clean decoupling between data sampling, model wrappers, inner optimizers, and loss modules.

---

## ⚠️ Computational Trade-offs: VRAM Consumption & Chunk Size

While vectorizing task execution via `vmap` provides theoretical and wall-clock speedups, it alters the memory scaling profile:

$$\text{Memory Overhead} \propto B_{\text{meta}} \times N_{\text{inner\_steps}} \times \text{Activation Size}$$

1. **Second-Order Derivatives & Activation Footprint:**  
   In higher-order optimization (e.g., Full MAML, ProtoMAML), computation graphs across all inner adaptation steps for all parallel tasks must reside in VRAM simultaneously. On consumer GPUs with limited VRAM, large meta-batch sizes can quickly lead to Out-Of-Memory (OOM) errors.

2. **Chunked Gradient Accumulation (`chunk_size`):**  
   To mitigate memory pressure, `Parallel-MetaLearn` implements chunked task processing (`chunk_size`). 
   * When `chunk_size` equals the meta-batch size, full vectorization is achieved.
   * If VRAM is constrained, decreasing `chunk_size` divides the meta-batch into smaller sub-batches and accumulates gradients sequentially. 
   * **Note:** In extreme scenarios where `chunk_size = 1`, memory usage drops to its minimum, but runtime performance converges to standard sequential iteration. Researchers should tune `chunk_size` to balance available hardware memory against parallelism throughput.

---

## Installation

### From PyPI
```bash
pip install parallel-metalearn

```

### For Local Development

```bash
git clone [https://github.com/your-username/parallel-metalearn.git](https://github.com/your-username/parallel-metalearn.git)
cd parallel-metalearn
pip install -e .

```

---

## Supported Algorithms

| Algorithm | Paradigm | Derivative Order | Key Reference |
| --- | --- | --- | --- |
| **MAML** | Gradient-based | 1st & 2nd Order | Finn et al. (2017) |
| **FOMAML** | Gradient-based | 1st Order | Finn et al. (2017) |
| **ANIL** | Representation-based | 1st & 2nd Order | Raghu et al. (2019) |
| **BOIL** | Body-Only Inner Loop | 1st & 2nd Order | Oh et al. (2020) |
| **Meta-SGD** | Learnable Step Sizes | 1st & 2nd Order | Li et al. (2017) |
| **MAML++** | Multi-Step Loss & MSL | 1st & 2nd Order | Antoniou et al. (2019) |
| **ProtoMAML (v1 & v2)** | Metric + Gradient Hybrid | 1st & 2nd Order | Triantafillou et al. (2019) |
| **Prototypical Networks** | Metric-based | Non-parametric | Snell et al. (2017) |
| **Reptile** | First-order Directional | 1st Order | Nichol et al. (2018) |

---

## Minimal Working Example

Below is a standard workflow demonstrating model initialization, loss configuration, and meta-training:

```python
import torch
from metalearn.model_wrappers import MAML_Model
from metalearn.loss import LabelEncoder, CrossEntropy, CategoricalAccuracy
from metalearn.inner_optimizers import InnerSGD
from metalearn.algorithms import MAML
from metalearn.train import MetaTrain

# 1. Standard PyTorch architecture definition
backbone = MyFeatureExtractor()
head = MyLinearClassifier()
model = MAML_Model(backbone=backbone, head=head)

# 2. Label encoding and loss setup
label_encoder = LabelEncoder(num_classes=10, max_n_way=3, shuffle=True)
loss_fn = CrossEntropy(metric_fn=CategoricalAccuracy())

# 3. Optimization setup
inner_optimizer = InnerSGD(
    initial_fast_weights=model.get_fast_weights(),
    inner_lr=0.01,
    first_order=False
)
outer_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Meta-Learner initialization
algorithm = MAML(
    model=model,
    optimizer=outer_optimizer,
    inner_optimizer=inner_optimizer,
    support_loss_fn=loss_fn,
    inner_steps=3,
    chunk_size=8,  # Balances VRAM overhead and vectorization speed
)

# 5. Training execution
trainer = MetaTrain(
    TrainLoader=train_loader,
    ValLoader=val_loader,
    algorithm=algorithm
)

history, best_metric, best_loss = trainer.train(
    epochs=100,
    check_idx=10,
    log_checkpoint_path="checkpoints"
)

```

---

## Research Applications & Extensions

The framework is decoupled via standardized input/output mappings (`out_dict`, `targets`), allowing straightforward application to various meta-learning paradigms:

1. **Multi-Task Meta-Learning (MTL):** Extend `targets` to return multiple supervisory signals and define composite objectives in `BaseLoss`.
2. **Domain Generalization & Shift:** Implement alignment objectives (e.g., MMD, Wasserstein loss) using features extracted from `out_dict["features"]`.
3. **Simulated Federated Meta-Learning:** Utilize `vmap` to execute localized client updates concurrently before applying server aggregation rules (e.g., FedAvg).
4. **Zero-Shot to Few-Shot Transition:** Models automatically switch from metric-based zero-shot priors to few-shot gradient adaptation depending on support set availability.


---
### 📊 Scaling Analysis: Vectorized (`vmap`) vs. Sequential (`for-loop`) Execution
---

To evaluate the empirical speedup and scaling profile of functional task vectorization, MAML was benchmarked across a wide spectrum of meta-batch sizes ($B_{\text{meta}} \in [1, 200]$) under identical architectural, loss, and optimization constraints. Each configuration was evaluated over 10 full meta-training epochs to compute the average execution latency per epoch.

| Meta-Batch Size (Tasks) | Sequential `for-loop` (ms/epoch) | Vectorized `vmap` (ms/epoch) | Speedup Factor |
| :---: | :---: | :---: | :---: |
| **1** | 79.43 ms | 260.29 ms | **0.31x** |
| **2** | 241.40 ms | 226.36 ms | **1.07x** |
| **3** | 86.57 ms | 104.86 ms | **0.83x** |
| **5** | 143.71 ms | 75.97 ms | **1.89x** |
| **10** | 253.73 ms | 79.87 ms | **3.18x** |
| **20** | 594.35 ms | 113.24 ms | **5.25x** |
| **30** | 760.44 ms | 149.40 ms | **5.09x** |
| **40** | 1063.92 ms | 224.18 ms | **4.75x** |
| **50** | 1555.70 ms | 232.85 ms | **6.68x** |
| **70** | 1863.73 ms | 305.81 ms | **6.09x** |
| **100** | 2920.59 ms | 386.20 ms | **7.56x** |
| **120** | 3259.52 ms | 447.91 ms | **7.28x** |
| **200** | 5490.08 ms | 791.53 ms | **6.94x** |

---

#### 🔍 Performance & Hardware Bottleneck Analysis

1. **Vectorization Overhead at Small Batches ($B_{\text{meta}} \le 3$):**  
   For very small task counts, the initial compilation and dispatch overhead of `torch.func` functional transformations dominates, resulting in lower throughput than native sequential iteration.

2. **Sub-linear Scaling & Core Occupancy ($B_{\text{meta}} = 5 \to 100$):**  
   As the number of concurrent tasks increases, `torch.func.vmap` maximizes Streaming Multiprocessor (SM) occupancy on the GPU. While the sequential execution latency grows strictly linearly ($\mathcal{O}(N)$), the vectorized pipeline scales sub-linearly, reaching a peak acceleration of **$\approx 7.56\times$** at 100 tasks.

3. **Speedup Saturation & Amdahl's Law ($B_{\text{meta}} > 100$):**  
   The empirical speedup plateaus between **$7\times$ and $7.5\times$** rather than scaling indefinitely. This saturation is governed by fundamental hardware constraints:
   * **Compute & Memory Bandwidth Saturation:** Once GPU CUDA cores reach full occupancy, additional tasks are queued by the hardware warp scheduler rather than executed with true instantaneous concurrency. Additionally, tracking multiple computation graphs under second-order derivatives shifts the bottleneck from compute throughput to GPU memory bandwidth.
   * **Amdahl's Law:** Non-vectorizable sequential operations (e.g., CPU data batching, host-to-device memory copies, outer-loop global parameter reduction, and outer optimizer updates) place an asymptotic upper bound on theoretical end-to-end acceleration.

4. **Hardware Context & Colab Constraints:**  
   > 💡 **Benchmark Hardware Note:**  
   > These benchmarks were conducted on a standard **free-tier Google Colab instance** (NVIDIA Tesla T4 GPU with ~15 GB VRAM). In this virtualized environment, physical GPU compute units and memory bandwidth are shared across multiple concurrent user sessions (typically allocating only a fraction of total hardware throughput to each runtime). On dedicated research-grade hardware (e.g., NVIDIA A100/H100 GPUs with high-bandwidth HBM3 memory), higher saturation thresholds and absolute throughput are expected.
---

## Empirical Benchmark (Fault Diagnosis Domain Shift)

To evaluate empirical convergence, algorithms were evaluated on the **CWRU Vibration Dataset** under strict file-level stratified partitioning (evaluating generalization under domain shift across distinct physical bearing loads).

### Setup

* **Signal Segmentation:** 2-channel vibration windows ($L=2048$, $75\%$ overlap).
* **Data Split:** $20\%$ of physical data files used for Meta-Training; $80\%$ reserved exclusively for Out-Of-Distribution Meta-Validation.
* **Task Protocol:** 3-Way 5-Shot Support ($K_s=5$), 15-Shot Query ($K_q=15$).
* **Batch Configuration:** Meta-Batch Size = $24$, evaluated over 200 epochs.

### Results

| Algorithm | Inner Loop Protocol | Peak Validation Accuracy | Empirical Characteristics |
| --- | --- | --- | --- |
| **ProtoMAML v2** | Prototypical Head + Adapted Backbone (3 Steps) | **100.00%** | Stable convergence; lower variance under domain shift. |
| **ProtoMAML v1** | Prototype Initialization + Joint SGD (1 Step) | **99.44%** | Fast adaptation; consistent loss minimization. |
| **MAML++** | Per-Layer LRs + Multi-Step Loss (3 Steps) | **98.89%** | Significant variance reduction over Vanilla MAML. |
| **Prototypical Net** | Non-parametric Distance Metric | **86.11%** | Fast computation; susceptible to representational underfitting. |
| **MAML (Vanilla)** | Second-Order SGD (3 Steps) | **82.22%** | Higher gradient variance across adaptation steps. |
| **Reptile** | First-Order Directional Update (3 Steps) | **70.56%** | Minimal VRAM footprint; requires more adaptation steps. |



## License

Distributed under the **MIT License**.
