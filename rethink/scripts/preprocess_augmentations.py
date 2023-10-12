import argparse
import os
import psutil
import multiprocessing
import torch

from rethink.preprocess import preprocess

def worker(preprocessing_config, csv_file, args):
    print(f"Preprocessing with {preprocessing_config}")

    temp = preprocessing_config.split("_")
    transform_name = "_".join(temp[2:])

    output_dir = os.path.join("/media/magalhaes/sound/spectograms", transform_name)
    print(output_dir)

    # If output dir exists and is not empty, skip
    if os.path.isdir(output_dir) and os.listdir(output_dir) and not args.replace:
        print("Skipping...")
        return

    preprocess(
        csv_file=csv_file,
        augmentations_file=os.path.join(args.configs_dir, preprocessing_config + ".json"),
        output_dir=output_dir,
        augment=True,
        device="cuda:0"
    )

    # After preprocessing, append the name of the augmentations_file to a log file
    with open("preprocessed_augmentations.log", "a") as f:
        f.write(preprocessing_config + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", type=str, default="./preprocessing_configs")
    parser.add_argument("csv_file", type=str)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    # List all files in the directory
    files = os.listdir(args.configs_dir)

    # Extract filenames without file extensions
    filenames_without_extension = [os.path.splitext(file)[0] for file in files]

    # Calculate the number of processes to use 70% of the system's memory
    system_memory = psutil.virtual_memory().total
    process_memory = 10 * 1024 * 1024 * 1024  # 10GB
    max_processes = int(0.7 * system_memory / process_memory)

    # Calculate the number of processes to use 80% of gpu memory, in VRAM, process takes about 2 GB
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    process_memory = 2 * 1024 * 1024 * 1024  # 2GB
    max_gpu_processes = int(0.8 * gpu_memory / process_memory)

    max_processes = min(max_processes, max_gpu_processes)

    # If num_workers is set, use that instead of the calculated value
    if args.num_workers:
        max_processes = args.num_workers
    
    print(f"Total system memory: {system_memory / (1024 ** 3)} GB")
    print(f"Maximum processes: {max_processes}")

    # If max_processes is 1, dont use multiprocessing
    if max_processes == 1:
        for config in filenames_without_extension:
            worker(config, args.csv_file, args)
        return

    # Create a pool of worker processes and distribute the work
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(worker, [(config, args.csv_file, args) for config in filenames_without_extension])

if __name__ == "__main__":
    main()
