import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


# Train model
def train_model(dataset, model, max_epochs, batch_size=1024, split_ratio=0.8, **dataloader_kwargs):
    # Calculate split sizes
    calculate_split_sizes = lambda dataset_size, split_ratio: (int(split_ratio * dataset_size), dataset_size - int(split_ratio * dataset_size))

    # Splitting the dataset
    train_size, eval_size = calculate_split_sizes(len(dataset), split_ratio)
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs)

    # Training the model
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)


def learn_docids(dataset, model, max_epochs, batch_size=1024, **dataloader_kwargs):
    # Creating dataloaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)

    # Training the model
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, dataloader)




