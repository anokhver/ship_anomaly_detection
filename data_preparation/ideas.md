For AIS trajectory anomaly detection with logistic regression, the issue you're describing is common - port areas have high natural variability that can confuse the model. Here are several data preparation strategies to address this:

## Spatial Preprocessing

**Port Area Masking**: Create geographical masks around known ports and either exclude these regions entirely or treat them separately. You can define circular or polygonal zones around ports where different anomaly thresholds apply.

**Distance-based Weighting**: Weight your features based on distance from ports - give less importance to variations when ships are within a certain radius of ports (e.g., 5-10 nautical miles).

## Feature Engineering

**Contextual Features**: Instead of just using raw position/speed data, create features that account for context:
- Distance to nearest port
- Time since departure/until arrival
- Whether the ship is in a "maneuvering zone" vs "open water"
- Speed relative to typical speeds in that area

**Trajectory Segmentation**: Split trajectories into phases:
- Port approach/departure (high variability expected)
- Open ocean transit (low variability expected)
- Coastal navigation
Train separate models or use different parameters for each phase.

## Statistical Normalization

**Local Standardization**: Instead of global z-score normalization, normalize features within geographical regions or trajectory phases. This prevents port area variability from dominating the feature space.

**Robust Scaling**: Use median and IQR instead of mean and standard deviation for scaling, as these are less sensitive to the extreme variations near ports.

## Sampling Strategy

**Stratified Sampling**: Ensure your training data has balanced representation from different trajectory phases, not just raw temporal sampling.

**Synthetic Minority Oversampling**: If true anomalies are rare, use techniques like SMOTE but applied spatially - generate synthetic anomalies in open water areas where they're more clearly detectable.

## Alternative Approach

Consider using **isolation forests** or **one-class SVM** instead of logistic regression for this type of problem, as they're specifically designed for anomaly detection and handle varying normal behavior better.

Would you like me to elaborate on any of these approaches or help you implement a specific preprocessing pipeline?