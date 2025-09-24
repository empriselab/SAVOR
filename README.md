# SAVOR: Skill Affordance Learning from Visuo-Haptic Perception for Robot-Assisted Bite Acquisition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A implementation for the paper [**SAVOR: Skill Affordance Learning from Visuo-Haptic Perception for Robot-Assisted Bite Acquisition**](https://emprise.cs.cornell.edu/savor/).

[Project Website](https://emprise.cs.cornell.edu/savor/) | [Paper](https://arxiv.org/abs/2506.02353)

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/empriselab/SAVOR.git
cd SAVOR
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up configuration:**
   - Add your OpenAI API key to `configs/openai_api_key.yaml` (if using LLM features)
   - The file should contain one line in YAML format: api_key: sk-xxxx

## 🚀 Quick Start

**Basic Testing:**
```bash
python test_basic.py
```

**Example Usage:**
```bash
python example_usage.py
```

**Training:**
```bash
python train.py --data_dir ./data
```

All LLM-query related code is located under the `utils/` directory.

## 📊 Data Structure

Our SAVOR dataset is stored in **RLDS** format. Dataset Link: [Link](https://drive.google.com/drive/folders/1CMU8bFS9Z7s76jVjZa-ISrv_0-RpxWlM?usp=sharing)

### Dataset Organization

```
data/
└── savor_rlds/
    └── 1.0.0/
        ├── dataset_info.json      # Dataset metadata
        ├── features.json          # Schema definition
        └── savor_rlds-train.tfrecord-00000-of-00001  # Data files
```

### Data Hierarchy

```
Episode
├── episode_metadata
│   ├── episode_id (int32)         # Unique episode identifier
│   ├── food_type (string)         # Type of food (e.g., "apple", "banana")
│   └── trajectory_id (int32)      # Trajectory number within food type
└── steps (sequence)
    ├── is_first (bool)            # True on first step of episode
    ├── is_last (bool)             # True on last step of episode
    ├── is_terminal (bool)         # True if episode ends
    ├── observation
    │   ├── rgb ([256,256,3])
    │   ├── depth ([256,256,1])
    │   ├── pose (float32[6])      # utensil pose [x,y,z,rx,ry,rz]
    │   └── force_torque (float32[6]) # Force/torque [Fx,Fy,Fz,Tx,Ty,Tz]
    └── physical_properties (float32[3]) # [softness, moisture, viscosity] (1-5 scale)
```

### Data Loading
```python
# Load RLDS dataset
ds = tfds.load('savor_rlds', data_dir=data_dir, split='train')

# Convert format
processor = SavorDataProcessor(data_dir=data_dir)
train_data = processor.get_data(split='train')
val_data = processor.get_data(split='val')
```

## 📚 Citation
```bibtex
@misc{wu2025savor,
  title={SAVOR: Skill Affordance Learning from Visuo-Haptic Perception for Robot-Assisted Bite Acquisition}, 
  author={Wu Zhanxin and Ai Bo and Silver Tom and Bhattacharjee Tapomayukh},
  year={2025},
  eprint={TODO},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
} 
```