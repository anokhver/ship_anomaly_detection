# Models

## Directory contents
- **📂 anylysis-distribution**
    - contains data analysis (semantics, types, distributions)
- **📂 labeled data**
    - contains parquet files with cleaned data, as well as trip data with partial manual labeling
    - contains **Anomaly_definition** with anomalies definition in ship trajectory data
- **📂 models_per_route**
    - contains model files for both routs with a dispatcher 
- **📂 LSTM**
    - contains all files connected to LSTM training
- **model_tuning.md**
    - describes what was done for tuning of the machine learning models
- **LSTM/README.md**
    - contains additional information about the LSTM autoencoder model used in the project


- **Anomaly_definition**
    - Self-explanatory
- **.py model files**
    - python files with shortened model names contain the training script for specific models

## **Selected Models**

1. **One-Class SVM (Distance-Based  - Semi-Supervised)**

    *Theory*
   - Learns **only normal behavior** during training. (Excludes anomalies from dataset)
   - Flags deviations from the learned normal pattern as anomalies.
   - Effective when anomalies are rare and training data is mostly normal.
   - Constructs a boundary around the normal data in a high-dimensional space; points outside are considered anomalous.

    *Observations during training*
   - It learned on the data confirmed to be normal. If there is new kind of normal data, it will mark it as an anomaly. Isolation forest is better in this regard.
   - High precision, but recall is lower than expected.
   - Correctly identifies anomalous sections of the trip. Marks a bit too much anomalies, needs tuning.
   - Potential tuning: change nu

2. **Isolation Forest (Isolation - Unsupervised)**

     *Theory*
   - Randomly partitions data; anomalies are isolated closer to the root.
   - Efficient for **high-dimensional data**, handles global anomalies well.
   - Based on the premise that anomalies are more susceptible to isolation due to their rarity and distinctness.
   - Does not rely on distance or density, making it robust to scaling and irrelevant features.

   *Observations during training*
   - Correctly identifies anomalous sections of the trip
   - Hard to balance between finding "too much" or "too little" anomalies
   - Able to find anomalies unseen in the training data
   - Potential tuning: balance n_estimators, contamination to find best possible precision and recall

3. **Random Forest (Classification - Supervised)**

    *Theory*
   - Data needs to be labeled
   - Uses an ensemble of decision trees to classify data; majority voting determines the final prediction.
   - Can detect anomalies if trained with both normal and anomalous labeled examples.
   - Robust to overfitting and performs well with a mix of feature types.

    *Observations during training*
   - Very high precision and recall on the labeled anomalies.
   - Correctly identifies anomalous sections of the trip.
   - Can struggle with types of anomalies it never saw before.
   - Potential tuning: change n_estimators and max_depth, proper balancing of class_weight

4. **Logistic regression (Classification - Supervised)**

    *Theory*
   - Data needs to be labeled
   - Models the probability of a binary outcome (e.g., normal vs anomaly).
   - Assumes a linear relationship between input features and the log-odds of the output.
   - Simple, interpretable, and effective when the dataset has clear linear separability.

   *Observations during training*
   - High precision, worse recall than random forest, especially on the Bremerhaven route.
   - Correctly identifies anomalous sections of the trip, though some points happen to fall below the scoring threshold.
   - Like random forest, struggles with anomalies before unseen.
   - Potential tuning: change in C (regularization strength), penalty, different solver algorithms, proper balancing of class_weight

5. **LSTM (Forecasting-Based - Semi-Supervised)**

    *Theory*
   - Learns to **reconstruct normal time series** through an encoder-decoder architecture; high reconstruction error indicates anomaly.
   - Captures temporal dependencies in sequential data, ideal for ship trajectory analysis.
   - Uses memory cells and gating mechanisms to retain long-term context and handle irregular time intervals.
   - Assumes anomalies significantly deviate from learned normal temporal patterns in reconstruction quality.

    *Observations during training*
   - Identifies most anomalies, but too many false positives (mostly because of ports)
   - We Need to incorporate flexible thresholding if the ship is in port (zones)
   - Potential tuning: adjust sequence length (currently 15), hidden size (64), threshold percentile, and number of layers for optimal performance, early stopping
   - Now overfitting, threshold calculation might be not optimal

## Model features
This is a summary of the data passed to the models during training. It is derived from the original data (after cleaning).

- **speed_over_ground**
    - Speed of the vessel at current time, SOG value from the source data
- **dv**
    - Delta of velocity - difference of speed between the current point and the previous one, derived from SOG
- **dcourse**
    - Delta of the course - difference of COG between the current point and the previous one, adjusted for angle calculations
- **ddraft**
    - Delta of the draught - difference of draught between the current point and the previous one
- **zone**
    - A pre-defined coordinate range, for which the models can differentiate between the "port" and "open sea" zones - in ports anomalous behaviour is supposed to be treated less harshly
- **x_km**
    - A substitute for evaluating coordinates - a "center of the route is established based on the coordinates of all entries in that route. Then, the distance from that center is calculated for an entry based on longitude difference and converted to kilometers.
- **y_km**
    - Same as x_km, only calculating latitude difference.
- **dist_to_ref**
    - Distance for each point to the average route. Average trajectory for a route is computed by resampling each trip to n_points along cumulative distance fraction, then averaging. 
- **route_dummy**
    - Scrapped idea, replaced by dispatcher files. Will be removed in sprint 4.