import os
import shutil
def main():
    # Ensure data_to_clean directory exists and clean it if needed
    data_dir = 'data_to_clean'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    elif os.path.isdir(data_dir):
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if files:
            print(f"Removing {len(files)} files from {data_dir}...")
            for f in files:
                os.remove(f)
            print("All files removed.")

    #copying
    input_dir = '../../../input_data'
    if os.path.isdir(input_dir):
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        if input_files:
            for f in input_files:
                shutil.copy(f, data_dir)
            print(f"Copied {len(input_files)} files from {input_dir} to {data_dir}.")
        else:
            print(f"No files found in {input_dir} to copy.")
    else:
        print(f"Input directory {input_dir} does not exist.")

    # Only proceed with data processing if we have files to process
    data_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    if not data_files:
        print("No data files to process. Pipeline terminated.")
        return

    import initial_cleaning as initial_data_processing
    initial_data_processing.merge_and_process_data()
    import normalization_pipeline as normalization_pipeline
    normalization_pipeline.destination_normalization()
    import noise_handling_pipeline as noise_handling_pipeline

    noise_handling_pipeline.noise_handling()


if __name__ == "__main__":
    main()
