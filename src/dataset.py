import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

class ToxicCommentsDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name='distilbert-base-uncased', max_length=128):
        """
        Initialize dataset with:
        - csv_path: path to cleaned parquet data
        - tokenizer_name: pretrained tokenizer to use
        - max_length: max token length for padding/truncation
        """
        # Load dataset efficiently using pandas parquet reader
        self.df = pd.read_parquet(csv_path)

        # Initialize Hugging Face tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

        # Store max_length parameter
        self.max_length = max_length

    def __len__(self):
        """Return number of samples (required by DataLoader)."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return one tokenized sample as dictionary:
        - input_ids: tensor of token IDs
        - attention_mask: tensor mask for real tokens vs padding
        - label: binary target
        """

        # Get comment text and label
        comment = str(self.df.iloc[idx]['comment_text'])
        label = self.df.iloc[idx]['target']

        # Tokenize comment text
        encoding = self.tokenizer(
            comment,
            truncation=True,        # Truncate to max_length if longer
            padding='max_length',   # Pad to max_length if shorter
            max_length=self.max_length,
            return_tensors='pt'     # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),         # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(), # [seq_len]
            'label': label
        }
