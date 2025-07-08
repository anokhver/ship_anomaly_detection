# LSTM Autoencoder training for Ship Trajectory Anomaly Detection
    
## Model Architectures

Two models were tested

### 1. LSTM Autoencoder (Lightweight)
```
Architecture:
├── Encoder LSTM
│   ├── Input Size: 4-5 features (varies by route)
│   ├── Hidden Size: 16
│   ├── Layers: 2 (stacked)
│   └── Dropout: 0.1
└── Decoder LSTM
    ├── Hidden Size: 16
    ├── Layers: 2 (stacked)
    ├── Output Layer: Linear(16 → input_size)
    └── Dropout: 0.1
```

### 2. Enhanced LSTM Autoencoder (Deep Architecture)

hidden size was tested from 8 to 32

```
Architecture:
├── Multi-Stage Encoder
│   ├── Stage 1: LSTM(input_size → hidden_size*2)
│   ├── Stage 2: LSTM(hidden_size*2 → hidden_size)
│   └── Bottleneck: Compressed latent representation
├── Bridge Layers
│   ├── Hidden Transform: Linear(hidden_size → hidden_size*2)
│   └── Cell Transform: Linear(hidden_size → hidden_size*2)
└── Multi-Stage Decoder
    ├── Stage 1: LSTM(hidden_size → hidden_size*2)
    ├── Stage 2: LSTM(hidden_size*2 → hidden_size*2)
    └── Output: Linear(hidden_size*2 → input_size)
```
## ___

**Lightweight vs Deep Architecture:**
The Deep Architecture achieved better results on validation data,\
reaching MCC and F1 scores as high as _0.89_, \
while the Lightweight model achieved _0.87_. \

However, when examined on unlabeled data manually and compared to the Random Forest model, \
both architectures showed signs of overfitting near the starting ports in the training data, \
incorrectly labeling normal departures from both Kiel and Bremerhaven routes as anomalies.

The complex model required noticeably more training time without proving to boost results substantially.

**Feature Engineering Limitations:** 
Adding zone features that boosted performance in other models did not help the LSTM autoencoder. \
This is because LSTM autoencoders use a fundamentally different approach to anomaly detection,\
focusing on sequence reconstruction error rather than feature-based classification.\ 
While zone features could theoretically be used to implement different thresholds for different geographical zones, \
time constraints prevented this exploration, and focus remained on improving the existing architecture.

**Overfitting Mitigation Attempts:** 
I attempted to reduce overfitting by adding a learning rate scheduler and changing the batch size in range from _16_ to _32_, \
but these adjustments provided minimal improvement.

**Model Selection Reasoning:** 
LSTM Autoencoders are data-hungry models that require substantial amounts of training data to generalize well.\
Given the limited labeled dataset available, 
the lightweight model proved more suitable for this task due to its reduced parameter count and faster training time. \
Additionally, since LSTM Autoencoders focus on encoding entire sequences rather than specific engineered features, \
they are inherently less sensitive to feature engineering compared to models like Random Forest, \
making the lightweight architecture more appropriate for our dataset constraints.

________

## Final Models setup 

### General Training Parameters

- **Optimizer**: AdamW (with weight decay for regularization)
- **Loss Function**: Mean Squared Error (MSE)
- **Anomaly Detection Threshold**: 95th percentile of reconstruction error
- **Architecture**: 2-layer LSTM with 16 hidden units per layer
- **Dropout**: 0.1
- **Sequence Length**: 15 time steps
- **Sequence Step**: 7 (overlapping windows)
- **Batch Size**: 32
- **Validation Split**: 10%

---

### Route-Specific Configurations

#### Kiel Route (Model K)

**Model Performance**:
- **Precision**: 0.8630
- **Recall**: 0.8956
- **F1**: 0.8790 
- **Mcc**: 0.8353

- **Features**:
- `speed_over_ground`
- `course_over_ground`
- `x_km` (projected coordinates)
- `y_km` (projected coordinates)
- `draught`
- `dv` (speed variation)

**Training Parameters**:
- **Learning Rate**: 0.005
- **Weight Decay**: 1e-3
- **Input/Output Size**: 6 features

---

#### Bremerhaven Route (Model B)

**Model Performance**
- **F1 Score**: 0.8073
- **Precision**: 0.8000  
- **Recall**: 0.8148
  - **MCC**: 0.7590
- 
**Features**:
- `speed_over_ground`
- `course_over_ground`
- `x_km` (projected coordinates)
- `y_km` (projected coordinates)
- `dist_to_ref` (distance to average route)
- `draught`
- `dv` (speed variation)

**Training Parameters**:
- **Learning Rate**: 0.001
- **Weight Decay**: 1e-5
- **Input/Output Size**: 7 features
---
