import sys
sys.path.append("./")  # Add project root to path

from src.train_loader import create_dataloader

if __name__ == "__main__":
    dataloader = create_dataloader('data/train_cleaned.parquet', batch_size=4, num_workers=0)

    for batch in dataloader:
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("Labels shape:", batch['label'].shape)
        break
