from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reranker:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set the model to the evaluation mode