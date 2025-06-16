import wandb

LEARNING_RATE =0.02
EPOCHS = 5

# Start a new wandb run to track this script.
wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mlx-gg",
    # Set the wandb project where this run will be logged.
    project="two-towers-gigglers",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Dual-encoder",
        "dataset": "MS/Marco",
        "epochs": EPOCHS,
    },
)