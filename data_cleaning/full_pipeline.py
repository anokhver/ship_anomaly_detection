
def main():
    import data_cleaning.initial_cleaning as initial_data_processing
    initial_data_processing.initial_data_processing()
    import data_cleaning.normalization_pipeline as normalization_pipeline
    normalization_pipeline.normalization()
    import data_cleaning.noise_handling_pipeline as noise_handling_pipeline
    noise_handling_pipeline.noise_handling()

if __name__ == "__main__":
    main()
