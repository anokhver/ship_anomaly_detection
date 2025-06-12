# Order in which the notebooks should be run:

1. `cleaning_pipeline.ipynb` connects steps of: 
   - `1_type_norm.ipynb` - normalizes the type of the data
   - `2_destination_norm.ipynb` - normalizes the destination column data naming
2. `noise_handling.ipynb` - fills missing values in the dataset, removes duplicates, and handles obvious noise in the data

**Note** \
Folder `redundant` contains notebooks that are not used in the current pipeline, \
but was used in the past.