sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'model_score',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-4,
            'max': 1e-2,
            'distribution': 'log_uniform'
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'margin': {
            'values': [0.1, 0.2, 0.5]
        },
        'epochs': {
            'value': 5
        }
    }
}
