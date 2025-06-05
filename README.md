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
takes prepared.parquet data and generates html file with analysis results.
  
### For outlier purgings