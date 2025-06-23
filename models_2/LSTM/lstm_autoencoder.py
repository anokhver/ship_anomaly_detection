import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class LSTMAutoencoder:
    """
    LSTM Autoencoder for vessel trajectory anomaly detection.

    This class encapsulates model creation, training, prediction, and evaluation
    for detecting anomalies in time series vessel trajectory data.
    """

    def __init__(self, sequence_length: int = 50, n_features: int = 9,
                 lstm_units: int = 64, dense_units: int = 32, dropout_rate: float = 0.2):
        """
        Initialize LSTM Autoencoder.

        Args:
            sequence_length: Number of time steps in each sequence
            n_features: Number of features per time step
            lstm_units: Number of LSTM units in encoder/decoder
            dense_units: Number of units in bottleneck layer
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        self.model = None
        self.scaler = None
        self.threshold = None
        self.history = None

    def build_model(self):
        """Build the LSTM autoencoder architecture."""
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.n_features))

        # Encoder
        encoded = LSTM(self.lstm_units, return_sequences=True)(input_layer)
        encoded = Dropout(self.dropout_rate)(encoded)
        encoded = LSTM(self.lstm_units // 2, return_sequences=False)(encoded)
        encoded = Dropout(self.dropout_rate)(encoded)

        # Bottleneck
        bottleneck = Dense(self.dense_units, activation='relu')(encoded)
        bottleneck = Dropout(self.dropout_rate)(bottleneck)

        # Decoder
        decoded = Dense(self.lstm_units // 2, activation='relu')(bottleneck)
        decoded = Dropout(self.dropout_rate)(decoded)
        decoded = RepeatVector(self.sequence_length)(decoded)
        decoded = LSTM(self.lstm_units // 2, return_sequences=True)(decoded)
        decoded = Dropout(self.dropout_rate)(decoded)
        decoded = LSTM(self.lstm_units, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.n_features))(decoded)

        # Create and compile model
        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy', 'precision', 'recall', 'auc'])

        return self.model

    def fit(self, X_train: np.ndarray, validation_data: Tuple[np.ndarray, np.ndarray] = None,
            epochs: int = 50, batch_size: int = 32, patience: int = 10):
        """
        Train the autoencoder on normal data only.

        Args:
            X_train: Training sequences (normal data only)
            validation_data: Tuple of (X_val, y_val) for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
        """
        if self.model is None:
            self.build_model()

        # Scale the data
        self.scaler = MinMaxScaler()
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, n_features)
        ).reshape(n_samples, n_timesteps, n_features)

        # Prepare validation data if provided
        X_val_scaled = None
        if validation_data is not None:
            X_val, y_val = validation_data
            n_val_samples = X_val.shape[0]
            X_val_scaled = self.scaler.transform(
                X_val.reshape(-1, n_features)
            ).reshape(n_val_samples, n_timesteps, n_features)

            # For autoencoder, only use normal validation data
            normal_mask = y_val == 0
            X_val_scaled = X_val_scaled[normal_mask]

        # Set up callbacks
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_loss', mode ='min'),

        ]

        # Train the model (autoencoder: input = output)
        validation_data_tuple = (X_val_scaled, X_val_scaled) if X_val_scaled is not None else None

        self.history = self.model.fit(
            X_train_scaled, X_train_scaled,
            validation_data=validation_data_tuple,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using reconstruction error.

        Args:
            X: Input sequences

        Returns:
            Tuple of (reconstruction_errors, predictions)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before prediction")

        # Scale the input data
        n_samples, n_timesteps, n_features = X.shape
        X_scaled = self.scaler.transform(
            X.reshape(-1, n_features)
        ).reshape(n_samples, n_timesteps, n_features)

        # Get reconstructions
        X_reconstructed = self.model.predict(X_scaled, verbose=0)

        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=(1, 2))

        # Set threshold if not set (using 95th percentile)
        if self.threshold is None:
            self.threshold = np.percentile(reconstruction_errors, 95)

        # Make predictions
        predictions = (reconstruction_errors > self.threshold).astype(int)

        return reconstruction_errors, predictions

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = None) -> Dict:
        """
        Evaluate the model performance.

        Args:
            X_test: Test sequences
            y_test: True labels
            threshold: Custom threshold for anomaly detection

        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        reconstruction_errors, _ = self.predict(X_test)

        # Use custom threshold if provided
        if threshold is not None:
            self.threshold = threshold

        y_pred = (reconstruction_errors > self.threshold).astype(int)

        # Calculate metrics
        metrics = {
            'threshold': self.threshold,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_test, reconstruction_errors),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'reconstruction_errors': reconstruction_errors.tolist(),
            'predictions': y_pred.tolist()
        }

        # Add ROC AUC if both classes are present
        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_test, reconstruction_errors)

        return metrics

    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_reconstruction_errors(self, reconstruction_errors: np.ndarray, y_test: np.ndarray):
        """Plot distribution of reconstruction errors."""
        plt.figure(figsize=(10, 6))

        normal_errors = reconstruction_errors[y_test == 0]
        anomaly_errors = reconstruction_errors[y_test == 1]

        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True)

        if self.threshold is not None:
            plt.axvline(self.threshold, color='red', linestyle='--', label=f'Threshold: {self.threshold:.4f}')

        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Example of how to use the class

    # Initialize the autoencoder
    autoencoder = LSTMAutoencoder(
        sequence_length=50,
        n_features=9,
        lstm_units=64,
        dense_units=32,
        dropout_rate=0.2
    )

    # Assume you have preprocessed sequences and labels
    # X_sequences: shape (n_samples, sequence_length, n_features)
    # y_labels: shape (n_samples,) with 0=normal, 1=anomaly

    # Split into train/test (use only normal data for training)
    # X_train = X_sequences[y_labels == 0]  # Only normal data for training
    # X_test, y_test = X_sequences, y_labels  # All data for testing

    # Train the model
    # autoencoder.fit(X_train, epochs=50, batch_size=32)

    # Make predictions
    # reconstruction_errors, predictions = autoencoder.predict(X_test)

    # Evaluate the model
    # metrics = autoencoder.evaluate(X_test, y_test)
    # print("Evaluation Metrics:", metrics)

    # Plot results
    # autoencoder.plot_training_history()
    # autoencoder.plot_reconstruction_errors(reconstruction_errors, y_test)