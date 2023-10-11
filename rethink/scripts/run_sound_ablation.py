import argparse
import os
import subprocess
import time
import multiprocessing

def train_and_test(config_file):
    print(f"Training and Testing with {config_file}")

    # Training command
    train_process = subprocess.Popen(["python", "train.py", "--config_path", config_file])
    train_process.wait()

    # Sleep for 30 seconds before testing
    print("Waiting for 30 seconds before testing...")
    time.sleep(30)

    # Testing command
    test_process = subprocess.Popen(["python", "evaluate.py", "--config_path", config_file])
    test_process.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=1, help="Number of simultaneous processes")
    args = parser.parse_args()

    # Directory where your configuration files are located
    config_dir = "sound_ablation_configs"

    # List of configuration files
    config_files = [os.path.join(config_dir, file) for file in os.listdir(config_dir) if file.endswith(".json")]

    # Create a pool of worker processes
    with multiprocessing.Pool(args.processes) as pool:
        pool.map(train_and_test, config_files)

if __name__ == "__main__":
    main()
