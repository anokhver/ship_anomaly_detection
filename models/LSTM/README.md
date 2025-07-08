# Directory Structure
```
├── data_preparation.py          # AIS data preprocessing pipeline for adding new fetures that are described in model directory README.md
├── final_training.ipynb         # Main training notebook with model implementations
├── lstm_encoder.py              # LSTM autoencoder model class
└── README.md                    # This file
```

## Training

Route-specific models: Separate LSTM autoencoders trained for different shipping routes (KIEL and BREMERHAVEN)

## Retraining on New Data
If you want to retrain the model on new labeled data, follow these steps:

1. **Data Preparation**: Run `data_preparation.py` with the proper `data_path = "....parquet"` pointing to your file containing labeled and cleaned trips. Ensure the data format is consistent with what the cleaning pipeline was built for. It will add few new features to the data that can be used for training. The output will be saved to `LSTM_preprocessed.parquet`.

2. **Model Training**: Execute `final_training.ipynb` to retrain the model. All necessary configurations are available in the notebook, and the trained model will be automatically saved to the `OUTPUT_DIR_AE` directory you configure.

### Requirements
- Labeled trip data in Parquet format
- Properly configured data path in `data_preparation.py`
- Set `OUTPUT_DIR_AE` parameter in `final_training.ipynb`

### Output Files
- `LSTM_preprocessed.parquet`: Preprocessed data with additional features
- Trained model saved to configured `OUTPUT_DIR_AE` directory

--- 

# LSTM theoretical notes

Maintains relevant information across time steps while forgetting irrelevant details. \
Uses gates to control what to remember, forget, and output.
The failure to reconstruct is detected by comparing the original input with they differ significantly, indicating an anomaly.


## Autoencoder vs Classifier
- **Classifier**: Takes input → predicts category ("this is a cat")
- **Autoencoder**: Takes input → recreates same input. If reconstruction fails, input is anomalous

**Classifiers** uses supervised learning principles, requiring labeled examples of both normal and anomalous behavior to learn decision boundaries between categories, will perform worst on unseen anomalies.\
In contrast, **autoencoders** employ unsupervised reconstruction learning, training exclusively on normal data to learn efficient representations and reconstruction patterns.

## Key Advantages of Autoencoders for Anomaly Detection

**1. Sensitivity to Anomalies**\
Unlike classifiers that can only detect anomaly types present in their training data, autoencoders can identify previously unseen anomalous patterns. When encountering trajectory behaviors that deviate from learned normal patterns, the reconstruction error naturally increases, providing a continuous measure of anomaly severity rather than discrete classification labels.

**2. Reconstruction Error as Anomaly Score**\
The reconstruction error provides a measure of how different an input is from normal patterns. Makes scoring more interpretable and clearer. 

**3. Efficient Feature Learning**\
Autoencoders automatically learn relevant features for trajectory reconstruction without requiring manual feature engineering. The encoder-decoder architecture naturally captures the most important aspects of normal vessel movement patterns, while the bottleneck layer forces the model to learn compact, meaningful representations of normal behavior.

## Encoder-Decoder Pipeline
1. **Encoder**: Compresses sequence into compact representation
2. **Bottleneck**: Dense summary of the pattern
3. **Decoder**: Reconstructs an original sequence from summary

LSTM has **cell state** (long-term memory) and **hidden state** (short-term memory).

### Three Gates Control Memory:

**Forget Gate**
- Decides what to remove from cell state
- Looks at previous hidden state + current input
- Outputs 0-1 for each memory bit (0=forget, 1=keep)

**Input Gate** 
- Decides what new info to store
- Creates candidate values to add
- Filters which candidates actually get stored

**Output Gate**
- Decides what parts of cell state to output
- Controls what becomes the new hidden state

### Memory Flow:
1. Forget gate removes irrelevant old memories
2. Input gate adds relevant new information  
3. Cell state = old memories (after forgetting) + new memories
4. Output gate decides what to expose from updated cell state


## Application to Maritime Trajectory Analysis

Research demonstrates the effectiveness of reconstruction-based approaches in maritime contexts.

The encoder-decoder pipeline proves particularly suited for trajectory data because it can:
- Compress complex multi-dimensional trajectory sequences into dense representations
- Reconstruct normal movement patterns with high fidelity
- Detect deviations in position, speed, and heading simultaneously
