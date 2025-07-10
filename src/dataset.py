import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

class ToxicCommentsDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name='distilbert-base-uncased', max_length=128):
        # Load dataset
        self.df = pd.read_parquet(csv_path)
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        comment = str(self.df.iloc[idx]['comment_text'])
        label = self.df.iloc[idx]['target']

        # Tokenize
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(),  # [seq_len]
            'label': label
        }