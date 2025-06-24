# Models

## Directory contents
- **📂 anylysis-distribution**
    - contains data analysis (semantics, types, distributions)
- **📂 labeled data**
    - contains parquet files with cleaned data, as well as trip data with partial manual labeling
- **Anomaly_definition**
    - Self-explanatory

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
   - Someobservation

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
   - Learns to **reconstruct normal time series**; high reconstruction error = anomaly.
   - Captures temporal dependencies, ideal for sequential data.
   - Uses memory cells and gating mechanisms to retain long-term context.
   - Assumes anomalies significantly deviate from learned temporal patterns.

   *Observations during training*
   - Someobservation
