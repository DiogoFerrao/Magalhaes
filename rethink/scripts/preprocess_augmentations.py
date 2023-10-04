import argparse
import os

from rethink.preprocess import preprocess

if __name__ == "__main__":

    # Receive arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", type=str, default="./preprocessing_configs")
    parser.add_argument("csv_file", type=str)
    args = parser.parse_args()

    
    # List all files in the directory
    files = os.listdir(args.configs_dir)

    # Extract filenames without file extensions
    filenames_without_extension = [os.path.splitext(file)[0] for file in files]

    # Preprocess the data for every preprocessing config in the dir
    for preprocessing_config in filenames_without_extension:
        print(f"Preprocessing with {preprocessing_config}")

        temp = preprocessing_config.split("_")
        transform_name = "_".join(temp[2:])

        output_dir = os.path.join(
            "/media/magalhaes/sound/spectograms",
            transform_name,
        )
        print(output_dir)
        # If output dir exists and is not empty, skip
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            print("Skipping...")
            continue
        preprocess(
            csv_file=args.csv_file,
            augmentations_file=os.path.join(args.configs_dir, preprocessing_config + ".json"),
            output_dir=output_dir,
            augment=True,
            device="cuda:0"
        )
