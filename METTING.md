### TODO

1. What speed is too much? 
2. Check if there is data on land
3. Do we fill in missing destination in between trips? (Not too important)


### Data Pre-processing [1]

1. **Remove unlikely data**:  
   - Data appearing on land.  
   - Speed greater than 30 knots or less than 0 knots.  
   - Other data standards as shown in Table 1.
    -MMSI 9 digits 
   
2. **Down-sampling**:  
   - Each trajectory was down-sampled at a sampling frequency of 3 minutes.

3. **Trajectory separation**:  
   - Trajectories that are too long or have too many data points are separated.  

---

### Encode 
- how to encode string?

### Labeling

- manual (with viewing of map) and competing
- clustering [1] or SVM or something like that

[1 Anomaly detection](https://www.sciencedirect.com/science/article/pii/S002980182303024X)