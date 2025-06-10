# Models

| Type             | Approach | Training Data Needed |
|------------------|----------|----------------------|
| **Unsupervised** | Detects anomalies based on frequency, shape, or distribution differences | None |
| **Supervised**   | Learns to classify normal vs. anomalous from labeled examples | Labeled (normal + anomalies) |
| **Semi-supervised** | Models only normal behavior, flags deviations | Only normal data |


| Method        | Approach                          | Anomaly Indicator               |
|--------------|-----------------------------------|----------------------------------|
| **Forecasting** | Predict next step(s), compare to actual | High prediction error            |
| **Distance**   | Measure neighbor distances (k-NN, k-Means) | Large distance to normal clusters |
| **Isolation**  | Random tree partitions (Isolation Forest) | Short isolation path in trees     |
| **Reconstruction** | Model normal data boundaries (One-Class SVM) | Fails to reconstruct/classify |


#### **Selected Models**  
1. **One-Class SVM (Distance-Based  - Semi-Supervised - TYPE III)**  
   - Learns **only normal behavior** during training. (Excludes anomalies from dataset)
   - Flags deviations from the learned normal pattern as anomalies.  
   - Effective when anomalies are rare and training data is mostly normal.  

2. **k-Means / k-NN (Distance-Based - Unsupervised - TYPE I)**  
   - Data needs to be labeled 
   - Uses **distance metrics** to detect rare/unusual subsequences.  
   - Anomalies have larger distances to neighbors.  
   - No training needed; works well for clustering-based anomalies.  

3. **Isolation Forest (Isolation - Unsupervised - TYPE I)**  
   - Randomly partitions data; anomalies are isolated closer to the root.  
   - Efficient for **high-dimensional data**, handles global anomalies well.  

4. **LSTM (Forecasting-Based - Semi-Supervised - TYPE III)**  
   - Learns to **reconstruct normal time series**; high reconstruction error = anomaly.  
   - Captures temporal dependencies, ideal for sequential data.  

### Evaluation metrics

---
### **References**  
- [Comprehensive Anomaly Detection Survey (HPI)](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2022_schmidl_anomaly.pdf)  
- [One-Class SVM Guide](https://spotintelligence.com/2024/05/27/anomaly-detection-one-class-svm/)  
- [LSTM Autoencoders Explained](https://medium.com/@zhonghong9998/anomaly-detection-in-time-series-data-using-lstm-autoencoders-51fd14946fa3)

# Data

The data is a time series. Simply put it captures how metrics change over time.
Given data is unlabeled, can have missing values and outliers.

**Definition** 2.1. A time series anomaly is a sequence of data points \
𝑇𝑖,𝑗 of length 𝑗 −𝑖 +1 ≥ 1 that deviates w. r. t. some characteristic em- \
bedding, model, and/or similarity measure from frequent patterns \
in the time series 𝑇

| Field              | Description                                               | Example/Format               | Units/Range       |
|--------------------|-----------------------------------------------------------|------------------------------|-------------------|
| **TripID**         | Unique ship identifier                                    | "TRP_123456"                 | -                 |
| **MMSI**           | Unique ship identifier (Maritime Mobile Service Identity) | 366982000              | 9-digit number    |
| **StartLatitude**  | Voyage start latitude                                     | 32.7104                      | Decimal degrees   |
| **StartLongitude** | Voyage start longitude                                    | -117.1735                    | Decimal degrees   |
| **StartTime**      | Voyage start timestamp                                    | 2024-05-20T14:30:00Z         | ISO 8601          |
| **EndLatitude**    | Voyage end latitude                                       | 35.4437                      | Decimal degrees   |
| **EndLongitude**   | Voyage end longitude                                      | 139.6380                     | Decimal degrees   |
| **EndTime**        | Voyage end timestamp                                      | 2024-05-25T08:15:00Z         | ISO 8601          |
| **StartPort**      | Origin port code/name                                     | "USLAX"                      | -                 |
| **EndPort**        | Destination port code/name                                | "JPTYO"                      | -                 |
| **ID**            | Internal record identifier                                | 789456123                    | Numeric/UUID      |
| **time**          | Timestamp of current position record                      | 2024-05-20T15:22:11Z         | ISO 8601          |
| **shiptype**      | Vessel category code                                      | 70 (cargo)                   | -                 |
| **Length**        | Ship length                                               | 200                          | Meters            |
| **Breadth**       | Ship width                                                | 32                           | Meters            |
| **Draught**       | Current underwater depth                                  | 12.5                         | Meters            |
| **Latitude**      | Real-time latitude position                               | 33.5412                      | Decimal degrees   |
| **Longitude**     | Real-time longitude position                              | -118.2543                    | Decimal degrees   |
| **SOG**           | Speed Over Ground                                         | 14.5                         | Knots (0-30+)     |
| **COG**           | Course Over Ground (true direction)                       | 215                          | Degrees (0-359)   |
| **TH**            | True Heading (ship orientation)                           | 220                          | Degrees (0-359)   |
| **Destination**   | Next planned port                                         | "PORT SINGAPORE"             | -                 |
| **Name**          | Ship's registered name                                    | "Ever Given"                 | -                 |
| **Callsign**      | Radio call sign                                           | "3ABC4"                      | -                 |
| **AisSource**     | Source of AIS data (satellite/terrestrial)                | "SAT"                        | "SAT"/"TER"       |

## Cleaned data
| Essential Field       | Purpose in Anomaly Detection               | Redundant/Removable Field | Reason for Removal              |
|-----------------------|-------------------------------------------|---------------------------|----------------------------------|
| **MMSI**              | Unique vessel identification              | -                         | Critical for tracking            |
| **Latitude/Longitude**| Real-time position points                 | Start/End Lat/Long         | Derived from trajectory points   |
| **time**             | Timestamp for each position               | StartTime/EndTime          | Calculated from sequence         |
| **SOG**              | Speed anomalies (e.g., sudden stops)      | -                         | Key behavioral metric            |
| **COG**              | Course deviations (e.g., off-route)       | -                         | Directional outliers             |
| **TH**               | Heading vs. COG mismatches                | -                         | Orientation anomalies            |
| _**shiptype**_         | Filter normal behavior by vessel class    | -                         | Context for movement patterns    |
| **Draught**          | Loading/unloading anomalies               | -                         | Weight-related deviations        |
| **AisSource**        | Data reliability flag                     | -                         | Gaps/errors in tracking          |

| **Redundant Fields**  | Why Remove?                               | Replacement Logic          |
|-----------------------|-------------------------------------------|----------------------------|
| TripID                | Duplicates MMSI + time range             | Use MMSI + timestamps      |
| StartPort/EndPort     | Infer from trajectory clustering         | Geofence coordinates       |
| Name/Callsign         | Identical to MMSI purpose                | MMSI is sufficient         |
| Length/Breadth        | Rarely affects trajectory patterns       | Keep only for specific models |


# Next sprint

- Decide on features that will determent if the data is anomaly
- Find missing data values and try to replace them (if the data for example has replacement logic), if impossible, remove them
- Clean data from unnecessary fields (quick)
- Label data 
- Deciding more attentively on metrics models will be evaluated
- Set up models (not training just a sceleton)