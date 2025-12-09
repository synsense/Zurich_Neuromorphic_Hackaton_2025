# Event-Based Eye Tracking Project

Build a novel eye tracking model using event cameras and deploy it on neuromorphic hardware ([Speck](https://www.synsense.ai/products/speck-2/)).

## Challenge Overview

**Goal**: Develop a **new** Spiking Neural Network (SNN) approach that:
1. Predicts pupil center coordinates from event camera data
2. Runs on SynSense Speck neuromorphic chip (or Specksim)
3. **Improves upon or innovates beyond existing solutions**

This is NOT about reproducing existing work. Use the provided resources as **starting points and references**, but bring your own ideas: new architectures, training strategies, data augmentation, hybrid approaches, etc.

**Evaluation**:
- Primary metric: Euclidean distance error (pixels) - lower is better
- Requirement: Model must be deployable on Speck/Specksim

## Datasets

You can use either or both datasets:

### Ini-30 Dataset
- 30 subjects with real DVS eye recordings
- 640x480 resolution, typically downsampled to 64x64 for Speck
- Labels: pupil center (x, y) coordinates
- Format: `.aedat4` event files
- **Download**: [Zenodo](https://zenodo.org/records/11203260)

### 3ET Synthetic Dataset
- Synthetic event-based eye tracking data
- 22 subjects with multiple video recordings
- Labels: pupil center (x, y) coordinates
- **Download (H5 event frames - recommended)**: [Google Drive](https://drive.google.com/drive/folders/16qH_wv_oVNysJARtHIUrIXbHjOygfq_i)
- **Download (AEDAT raw + video)**: [Google Drive](https://drive.google.com/drive/folders/1HeOS5YBLruzHjwMKyBQfVTc_mJbsy_R1)

After downloading, set your data paths in `.env` (copy from `.env.example`).

## Speck Hardware Constraints

Your model **must** satisfy these constraints to run on Speck:

| Constraint | Requirement |
|------------|-------------|
| Neuron type | IAF (Integrate-and-Fire) only |
| Activation | No ReLU/Sigmoid - use spiking neurons |
| Pooling | Sum pooling only (no Max/Avg) |
| Weights | 8-bit quantized |
| Neuron states | 16-bit |
| Max layers | 9 convolutional layers |
| Max input size | 128x128 pixels |
| Max channels | 1024 per layer |

Use `sinabs` to validate your model:
```python
from sinabs.backend.dynapcnn import DynapcnnNetwork

dynapcnn_net = DynapcnnNetwork(snn=your_model, input_shape=(2, 64, 64), discretize=True)
dynapcnn_net.make_config(device="speck2fmodule")  # Raises error if model doesn't fit
```

## Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/synsense/Zurich_Neuromorphic_Hackaton_2025.git
cd Zurich_Neuromorphic_Hackaton_2025/Speck_project

conda create -n eye-tracking python=3.10 -y
conda activate eye-tracking
pip install -r requirements.txt
```

### 2. Configure Data Paths

```bash
cp .env.example .env
# Edit .env with your dataset paths
```

### 3. Explore the Tutorial

Open `tutorial.ipynb` to load and visualize the data, understand the problem, and see Speck constraints in action.

### 4. Build Your Solution

**Reference implementations** (starting points, not solutions to copy):
- [RETINA](https://github.com/pbonazzi/retina) - SNN achieving 3.24 px error, deployable on Speck
- [3ET ConvLSTM](https://github.com/qinche106/cb-convlstm-eyetracking) - ConvLSTM baseline (not Speck-compatible, needs conversion)

**Sinabs tutorials** (for learning SNN training and Speck deployment):
- [Training SNNs with BPTT](https://sinabs.readthedocs.io/v3.1.1/tutorials/bptt.html)
- [ANN to SNN Conversion](https://sinabs.readthedocs.io/v3.1.1/tutorials/weight_transfer_mnist.html)
- [Speck Deployment (N-MNIST example)](https://sinabs.readthedocs.io/v3.1.1/tutorials/nir_to_speck.html#deploy-the-model-onto-speck)
- [Specksim](https://sinabs.readthedocs.io/v3.1.1/speck/specksim.html) - test without hardware

## Ideas to Explore

Here are some directions (not exhaustive):

- **Architectures**: Attention mechanisms, skip connections, novel temporal processing
- **Training**: Different surrogate gradients, knowledge distillation, data augmentation for events
- **Representations**: Time surfaces, learned encodings, adaptive temporal windowing
- **Hybrid approaches**: Combining both datasets, transfer learning

## Submission Requirements

1. **Working code** that trains and evaluates your model
2. **Trained model** that predicts (x, y) pupil coordinates
3. **Speck-compatible architecture** (validated with sinabs)
4. **Brief description** of your approach and what's novel about it

**Evaluation metric**:
```python
import numpy as np

def euclidean_error(predictions, targets):
    """Mean Euclidean distance in pixels (lower is better)"""
    return np.mean(np.sqrt(np.sum((predictions - targets) ** 2, axis=1)))
```

## Resources

- [Sinabs Documentation](https://sinabs.readthedocs.io/v3.1.1/) - SNN library for Speck
- [Tonic](https://tonic.readthedocs.io/) - Event camera data loading
- [RETINA Paper](https://arxiv.org/abs/2312.00425)
- [3ET Paper](https://arxiv.org/abs/2308.11771)

## Tips

1. **Validate Speck compatibility early** - don't train a model that won't fit
2. **Start simple, iterate fast** - get something working, then improve
3. **Use 64x64 input** - standard for Speck deployment
4. **Be creative** - the best solutions often come from unexpected directions

---

Good luck and have fun!

