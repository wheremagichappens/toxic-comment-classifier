import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent parallelism warning on Mac

from torch.utils.data import DataLoader
from src.dataset import ToxicCommentsDataset  # Absolute import best practice

def create_dataloader(csv_path, batch_size=32, max_length=128, shuffle=True, num_workers=0):
    """
    Create DataLoader for ToxicCommentsDataset.

    Parameters:
    - csv_path: path to cleaned data
    - batch_size: number of samples per batch
    - max_length: max token length for tokenizer
    - shuffle: whether to shuffle data each epoch
    - num_workers: parallel data loading workers

    Returns:
    - PyTorch DataLoader object
    """
    dataset = ToxicCommentsDataset(csv_path, max_length=max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
        # Default collate_fn works since dataset returns dicts of tensors with same shapes
    )

    return dataloader
