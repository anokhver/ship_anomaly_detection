- [ ] Run clustering on separated trips by start port and together too (pipeline with be the same just load dataframe 'fix_noise.parquet')
- [ ] Please create a function to visualise the trips and mark them different colors based on what cluster they belong to
- [ ] See if it clustered together trips that were marked manually as the same type (anomaly, normal)

### Clustering Basic concept
Clustering is a type of unsupervised machine learning that involves grouping a set of data points into clusters.
For this we do not need labels, as it finds the patterns byits own. \
Some algorithms require to specify the number of clusters, while others do not. 

For vessel trajectory analysis, clustering can help identify: 
- Common/normal route patterns
- Anomalous trajectories that deviate from normal patterns
- Different route types used by vessels

### What type of clustering you can try
[Ship Anomalous Behavior Detection Using Clustering](https://www.mdpi.com/2077-1312/11/4/763) \
[An anomaly detection method based on ship behavior trajectory](https://www.sciencedirect.com/science/article/pii/S002980182303024X)
