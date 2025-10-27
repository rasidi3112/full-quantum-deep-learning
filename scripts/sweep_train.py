import wandb # type: ignore

def train():
    wandb.init()
    config = wandb.config
   
    train_loader, val_loader = load_qm9(batch_size=config.batch_size) # type: ignore
    
    trainer = QuantumTrainer(config, train_loader, val_loader) # type: ignore
    trainer.fit()

sweep_id = wandb.sweep("sweep.yaml", project="full-quantum-dl")
wandb.agent(sweep_id, function=train, count=10)
