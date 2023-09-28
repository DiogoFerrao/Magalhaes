from rethink.preprocess import preprocess
import argparse
import os

if __name__ == "__main__":
    
    # Receive arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", type=str, default="./preprocessing_configs")
    parser.add_argument("csv_file", type=str)
    args = parser.parse_args()

    # Preprocess the data for every preprocessing config in the dir
    for preprocessing_config in os.listdir(args.configs_dir):
        if preprocessing_config.endswith(".json"):
            print(f"Preprocessing with {preprocessing_config}")

            temp = preprocessing_config.rstrip(".json").split("_")

            output_dir = os.path.join(
                "/media/magalhaes/sound/spectograms",
                temp[2]+temp[3],
            )
            print(output_dir)
            preprocess(
                csv_file=args.csv_file,
                augmentations_file=os.path.join(args.configs_dir, preprocessing_config),
                output_dir=output_dir,
                augment=True,
                device="cuda:0"
            )
