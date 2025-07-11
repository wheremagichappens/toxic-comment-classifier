from transformers import DistilBertForSequenceClassification

def load_model(num_labels=2, freeze_bert=False):
    """
    Load DistilBERT model for sequence classification.

    Parameters:
    - num_labels: number of output classes (2 for binary classification)
    - freeze_bert: whether to freeze DistilBERT encoder weights

    Returns:
    - model: DistilBERT sequence classification model
    """
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    if freeze_bert:
        for param in model.distilbert.parameters():
            param.requires_grad = False

    return model
