import os
import pandas as pd
import yaml
from pytorch_lightning import Trainer, seed_everything
import importlib 
import argparse

results_df = pd.DataFrame()
seed_everything(42)


parser = argparse.ArgumentParser(description='Automate testing of trained models.')
parser.add_argument('model_dirs', nargs='+', help='List of directories where models are stored.')
parser.add_argument('--csv', type=str, help='Path to an existing CSV file to append results to.')
parser.add_argument('--result_name', type=str, help='Name of the resulting DataFrame CSV file.')
parser.add_argument('--batch_size', type=int, help='Batch size for testing. Default: Use specified.')

args = parser.parse_args()

if args.csv and os.path.exists(args.csv):
    results_df = pd.read_csv(args.csv)
else:
    results_df = pd.DataFrame()

# Function to find the checkpoint with the lowest MSE loss
def get_lowest_mse_checkpoint(checkpoints_folder):
    checkpoints = [
        f for f in os.listdir(checkpoints_folder)
        if f.endswith('.ckpt') and 'mse_loss=' in f
    ]
    mse_losses = [
        (f, float(f.split('mse_loss=')[1].split('.ckpt')[0]))
        for f in checkpoints
    ]
    mse_losses.sort(key=lambda x: x[1])
    return os.path.join(checkpoints_folder, mse_losses[0][0])

def dynamic_class_import(module_name, class_name):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

for model_dir in args.model_dirs:
    for version_folder in os.listdir(model_dir):
        version_path = os.path.join(model_dir, version_folder)
        if os.path.isdir(version_path) and 'checkpoints' in os.listdir(version_path):
            checkpoints_folder = os.path.join(version_path, 'checkpoints')
            
            # Get the checkpoint with the lowest MSE loss
            config = yaml.safe_load(open(os.path.join(version_path, 'config.yaml')))
            lowest_mse_checkpoint = get_lowest_mse_checkpoint(checkpoints_folder)

            # Load the model from the checkpoint
            model_class = dynamic_class_import(*config["model"]["class_path"].split("."))

            if "dict_kwargs" in config["model"]:
                model = model_class(**config["model"]["init_args"], **config["model"]["dict_kwargs"])
            else:
                model = model_class(**config["model"]["init_args"])

            data_module_class = dynamic_class_import(*config["data"]["class_path"].split("."))
            if args.batch_size is not None:
                config["data"]["init_args"]["batch_size"] = args.batch_size
            if "dict_kwargs" in config["data"]:
                data_module = data_module_class(**config["data"]["init_args"], **config["data"]["dict_kwargs"])
            else:
                data_module = data_module_class(**config["data"]["init_args"])


            if "plugins" in config["trainer"]:
                del config["trainer"]["plugins"]
            if "callbacks" in config["trainer"]:
                del config["trainer"]["callbacks"]
            config["trainer"]["default_root_dir"] = "/scratch/tk2801/LightningT5/lightning_logs/test"
            # Initialize a trainer


            trainer = Trainer(**config["trainer"])

            # Run the testing procedure
            test_results = trainer.test(model=model, datamodule=data_module, ckpt_path=lowest_mse_checkpoint, verbose=True)



            # Append the results to the results_df DataFrame
            indiv_df = pd.DataFrame(test_results)
            indiv_df["version"] = version_folder.split("_")[-1]
            indiv_df["model class"] = config["model"]["class_path"].split(".")[-1]
            indiv_df["experiment name"] = os.path.basename(os.path.dirname(model_dir))
            results_df = pd.concat([results_df, indiv_df], ignore_index=True)

# Save the results_df DataFrame to a CSV file (optional)

if args.result_name is not None:
    results_df.to_csv(args.result_name, index=False)
elif args.csv:
    results_df.to_csv(args.csv, index=False)
else:
    results_df.to_csv('testings_results.csv', index=False)

# Display the results_df DataFrame
print(results_df)
