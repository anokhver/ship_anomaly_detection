## LSTM

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
