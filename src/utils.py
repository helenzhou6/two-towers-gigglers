import torch
import wandb

def init_wandb(lr = None, epochs = None):
    # Start a new wandb run to track this script.
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="mlx-gg",
        # Set the wandb project where this run will be logged.
        project="two-towers-gigglers",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "Dual-encoder",
            "dataset": "MS/Marco",
            "epochs": epochs,
        },
    )

def save_model(model_name, model_description, file_extension = 'pt'):
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=model_description
    )
    artifact.add_file(f"./data/{model_name}.{file_extension}")
    wandb.log_artifact(artifact)

def load_model(model_name):
    downloaded_model_path = wandb.use_model(model_name)
    return downloaded_model_path

def get_device():
    if (torch.cuda.is_available()):
        return "cuda"
    elif (torch.backends.mps.is_available()):
        return "mps"
    else:
        return "cpu"
