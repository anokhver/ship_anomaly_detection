## Directory Structure
```
├── data_preparation.py          # AIS data preprocessing pipeline for adding new fetures that are described in model directory README.md
├── final_training.ipynb         # Main training notebook with model implementations
├── lstm_encoder.py              # LSTM autoencoder model class
├── LSTM_explanation.md          # Theoreticall description of LSTM 
└── README.md                    # This file
```

## Training

Route-specific models: Separate LSTM autoencoders trained for different shipping routes (KIEL and BREMERHAVEN)

### Retraining on New Data
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

