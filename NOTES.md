kiel format?????

- do not transform the types of columns 
- use proper types True, False

df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')


# Models

## Selected Models  

1**k-Means (Distance-Based - Unsupervised - TYPE I)**  
   - Data needs to be labeled 
   - Uses **distance metrics** to detect rare/unusual subsequences.  
   - Anomalies have larger distances to neighbors.  
   - No training needed; works well for clustering-based anomalies.  

2**Isolation Forest (Isolation - Unsupervised - TYPE I)**  
   - Randomly partitions data; anomalies are isolated closer to the root.  
   - Efficient for **high-dimensional data**, handles global anomalies well.  

3**One-Class SVM (Distance-Based  - Semi-Supervised - TYPE III)**
   **Training: Only normal data**
   - Learns **only normal behavior** during training. (Excludes anomalies from dataset)
   - Flags deviations from the learned normal pattern as anomalies.  
   - Effective when anomalies are rare and training data is mostly normal.  

4**LSTM (Forecasting-Based - Semi-Supervised - TYPE III)**  
   - Learns to **reconstruct normal time series**; high reconstruction error = anomaly.  
   - Captures temporal dependencies, ideal for sequential data.  

---
### References  
- [One-Class SVM Guide](https://spotintelligence.com/2024/05/27/anomaly-detection-one-class-svm/)  
- [LSTM Autoencoders Explained](https://medium.com/@zhonghong9998/anomaly-detection-in-time-series-data-using-lstm-autoencoders-51fd14946fa3)