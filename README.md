# Deep Learning-Based Indoor Localization Using BLE Wearable Measurements


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)

Official implementation of the paper: **"Deep Learning-Based Indoor Localization Using BLE Wearable Measurements: A Multi-Environment Evaluation with Real-Time Deployment Validation"**



Key Features

- Deep Learning Architectures 1D CNN, Bi-LSTM, GRU, Transformer, Hybrid CNN-LSTM, Attention-LSTM
- Classical Baselines**: Random Forest, KNN, SVM, Decision Tree, Gradient Boosting, Naive Bayes, Linear Regression
- **Multi-Environment Evaluation**: Tested across 2 distinct indoor environments (31,450 samples)
- **ESP32-S3 Deployment**: Real-time inference on embedded hardware with measured latency and energy consumption
- **Transfer Learning**: Framework for adapting to new environments with only 30% labeled data
- **Complete Reproducibility**: All hyperparameters, random seeds, and training configurations documented


### Installation

```bash
# Clone the repository
git clone https://github.com/ble-localization/deep-learning-wearable-v2.git
cd deep-learning-wearable-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

bash
# Train LSTM model on Environment A
python train.py --model lstm --dataset data/environment_a --epochs 100

# Train with specific random seed for reproducibility
python train.py --model lstm --dataset data/environment_a --seed 42

# Train all models (5 runs each with different seeds)
bash scripts/train_all_models.sh


Evaluation

bash
# Evaluate trained model
python evaluate.py --model checkpoints/lstm_best.pth --dataset data/environment_a

# Cross-environment evaluation
python evaluate.py --model checkpoints/lstm_envA.pth --dataset data/environment_b --mode zero_shot

# Generate confusion matrix and plots
python evaluate.py --model checkpoints/lstm_best.pth --dataset data/environment_a --visualize
```

Transfer Learning

bash
# Fine-tune on new environment with 30% data
python transfer_learning.py \
  --pretrained checkpoints/lstm_envA.pth \
  --target_dataset data/environment_b \
  --labeled_ratio 0.3 \
  --epochs 50

 Repository Structure


deep-learning-wearable-v2/
├── data/
│   ├── environment_a/          # Environment A dataset (Zenodo)
│   ├── environment_b/          # Environment B dataset (contact authors)
│   └── preprocessing/          # Preprocessing scripts
├── models/
│   ├── cnn.py                  # 1D CNN architecture
│   ├── lstm.py                 # Bidirectional LSTM
│   ├── gru.py                  # GRU architecture
│   ├── transformer.py          # Transformer encoder
│   ├── cnn_lstm.py             # Hybrid CNN-LSTM
│   └── attention_lstm.py       # Attention-augmented LSTM
├── preprocessing/
│   ├── imputation.py           # KNN imputation (k=3)
│   ├── smoothing.py            # Savitzky-Golay filter
│   ├── normalization.py        # Min-max normalization
│   └── windowing.py            # Temporal windowing
├── training/
│   ├── train.py                # Main training script
│   ├── trainer.py              # Training loop implementation
│   └── config.py               # Hyperparameter configurations
├── evaluation/
│   ├── evaluate.py             # Evaluation script
│   ├── metrics.py              # Accuracy, error, confusion matrix
│   └── visualization.py        # Plotting functions
├── transfer_learning/
│   ├── transfer_learning.py    # Transfer learning framework
│   └── fine_tune.py            # Fine-tuning utilities
├── esp32_deployment/
│   ├── firmware/               # ESP32-S3 firmware code
│   ├── model_conversion/       # PyTorch → TFLite conversion
│   ├── inference_engine/       # C++ inference code
│   └── benchmarking/           # Latency/energy measurement tools
├── experiments/
│   ├── ablation_studies.py     # Ablation experiment scripts
│   ├── body_shadowing.py       # Body shadowing quantification
│   └── beacon_visibility.py    # Varying beacon visibility tests
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_analysis.ipynb
│   └── 05_esp32_deployment.ipynb
├── scripts/
│   ├── train_all_models.sh     # Train all models with 5 seeds
│   ├── run_ablations.sh        # Run all ablation studies
│   └── generate_figures.py     # Generate paper figures
├── checkpoints/
│   ├── pretrained/             # Pre-trained models (Environment A)
│   └── best_models/            # Best performing models
├── results/
│   ├── training_logs/          # TensorBoard logs
│   ├── evaluation_results/     # Evaluation metrics
│   └── figures/                # Generated figures
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 💾 Datasets

### Environment A
- **Source**: Publicly available on Zenodo
- **DOI**: [10.5281/zenodo.13317046](https://doi.org/10.5281/zenodo.13317046)
- **Samples**: 21,975 measurements
- **Locations**: 10 distinct locations
- **Beacons**: 10 BLE beacons
- **Download**: Automatically downloaded by data loading scripts

### Environment B
- **Samples**: 9,475 measurements
- **Locations**: 8 distinct locations
- **Beacons**: 12 BLE beacons
- **Access**: Available upon reasonable request to corresponding author
- **Note**: Different floor plan (L-shaped), ceiling-mounted beacons, concrete construction

## Technical Details

### Preprocessing Pipeline

1. **KNN Imputation** (`k=3`)
   - Handles missing beacon measurements
   - Euclidean distance metric

2. **Savitzky-Golay Smoothing** (`window=5, order=2`)
   - Reduces noise while preserving trends
   - Applied to each beacon's RSSI time series

3. **Min-Max Normalization** (`range=[-1, 1]`)
   - Stabilizes neural network training
   - Computed per beacon across training set

4. **Temporal Windowing** (`window=10, overlap=50%`)
   - Creates sequences for temporal models
   - Captures signal dynamics

Model Architectures

All models implemented in PyTorch with identical training configurations:
-Optimizer: Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸)
-Learning Rate: 0.001 (reduced by 0.5 on plateau, minimum 10⁻⁶)
-Batch Size: 64
-Early Stopping: Patience 20 epochs on validation loss
-Regularization: Dropout (0.3 hidden, 0.5 output), Weight decay 10⁻⁴
-Loss: Cross-entropy with class weights

ESP32-S3 Deployment
Hardware: ESP32-S3 DevKitC-1
- Processor: Xtensa LX7 dual-core @ 240 MHz
- Memory: 8MB PSRAM, 512KB SRAM
- Power: 3.3V supply
Software Stack:
- Framework: ESP-IDF v5.1
- Inference: TensorFlow Lite 2.13.0
- Optimization: XTensa NN kernels

Conversion Pipeline:

PyTorch (.pth) → ONNX → TensorFlow → TensorFlow Lite (float32)


Reproducibility

All experiments use fixed random seeds for reproducibility:
- Seeds: `{42, 123, 456, 789, 1011}`
- 5 independent runs per configuration
- Statistical significance tested with paired t-tests (p<0.01)
- Effect sizes reported using Cohen's d

Example: Exact Reproduction

bash
# Reproduce LSTM results from paper
python train.py \
  --model lstm \
  --dataset data/environment_a \
  --seed 42 \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --patience 20

Expected output:
Final Validation Accuracy: 82.1% ± 0.8%
Mean Localization Error: 2.3m ± 0.15m


Experiments

Ablation Studies

bash
# Run all preprocessing ablations
python experiments/ablation_studies.py --type preprocessing

# Run all architecture ablations
python experiments/ablation_studies.py --type architecture

Generate ablation results table
python experiments/ablation_studies.py --summarize

Body Shadowing Quantification
bash
Quantify body shadowing effects
python experiments/body_shadowing.py \
  --orientations 8 \
  --positions 4 \
  --duration 60

Results: Mean attenuation 12.3±3.2 dB


ESP32 Deployment Guide

Detailed deployment instructions in `esp32_deployment/README.md`

Quick Deploy:
```bash
# Convert trained model to TFLite
cd esp32_deployment/model_conversion
python convert_to_tflite.py --model ../../checkpoints/lstm_best.pth

# Flash firmware to ESP32
cd ../firmware
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Citation

If you use this code in your research, please cite:
bibtex
@article{yourname2024ble,
  title={Deep Learning-Based Indoor Localization Using BLE Wearable Measurements: A Multi-Environment Evaluation with Real-Time Deployment Validation},
  author={[Your Names]},
  journal={[Journal Name]},
  year={2024}
}

Contact

For questions about the paper or code:
- Email: [mohammed.shifaa25@ntu.edu.iq]
- Issues: [GitHub Issues](https://github.com/ble-localization/deep-learning-wearable-v2/issues)


Acknowledgments

- Dataset Environment A provided by Baejah et al. ([Zenodo](https://doi.org/10.5281/zenodo.13317046))
- Funding: NSF Grants CNS-2134567 and CNS-2245123


Related Work

- [BLE Indoor Localization Survey](https://ieeexplore.ieee.org/)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-S3 Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/)

