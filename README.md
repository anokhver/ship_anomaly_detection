### Get data for outlier purging

Refer to `data_cleaning` directory

- Run `cleaning_pipeline.py`  or `cleaning_pipeline.ipynb.` They are identical in functionality.
- For detailed step by step research for data cleaning go to notebooks folder.
- Or load **prepared.parquet**: contains prepared data for outliers purging 
    - normalized data types
    - consistent destination naming
    - missing values filled properly
    - AirSource column dropped during missing values handling

### Run the analysis, 

Refer to the `analysis` directory

`data_proccesing.py`: \
takes raw cvs data and generates html file with analysis results.  
And saves new cvs file with dropped attributes

`generate_analysis_parquet.py`: \
takes `prepared.parquet` data and generates html file with analysis results.

# For outlier detection
Read attentively all notebook if you want to change the outlier detection process.
**Output**: `fix_noise.parquet`

The main notebook for it is noise_handling.ipynb. It contains the following steps:
0. **Load data**: Load the prepared.parquet file.
1. **Convert to null** : 
   - Length: Converted to null when <= 0. 
   - Breadth: Converted to null when <= 0. 
   - Draught: Converted to null when <= 0. (can be 0.1 maybe, so it is not converted to null) 
   - Shiptype: Converted to null when <= 0. (after looking into data it's missing when lenght and 
     breadth are missing so assume its a type when we don't know ship type)
   - TH: Converted to null when the value is outside range <0;360> (inclusive) and not equal to 511. (511 is a AIS regular value for when we don't know, is not noise) 
   - COG: Converted to null when the value is outside range <0;360> (inclusive)
2. **Fill in Distination**
    - If null find the closest destination not null value for each ship top and bottom
    - Compare the coordinates of those two entries and fill in null with the one that has closer coordinates 
3 **Length && Breadth && Shiptype** 
    - Only fill in if there are at least one not null value
    - They should be static, so we just fill in with the most common value for each ship (if it is not null)
4 **Draught** 
    - Only fill in if there are at least one not null value
    - After analysis fill in with the median of a trip
5 **COG**
    - Was only one null last entry for trip filled with 0
6 **Handel values that are 0 for whole trip**
    - Train a HistGradientBoostingRegressor model to predict missing Draught values based on Length and Breadth
  
| Parameter     | Missing % |
|---------------|-----------|
| Length        | 1.122565  |
| Breadth       | 1.122565  |
| Draught       | 1.784436  |
| COG           | 0.000109  |
| Destination   | 0.575840  |
| COG           | 0.000109  |
| shiptype      | 0.717820  |


After

| Parameter     | Missing % |
|---------------|-----------|
| Destination   | 0.007443  |
| shiptype      | 0.717820  |

**TODO**
What to do with trips where:
-Length && Breadth && Shiptype all null
-Destination is all null
-Draught is all null