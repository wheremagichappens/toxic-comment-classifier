import sys
sys.path.append("./")  # Adds project root to path

from src.model import load_model

if __name__ == "__main__":
    model = load_model(num_labels=2, freeze_bert=False)
    print(model)

