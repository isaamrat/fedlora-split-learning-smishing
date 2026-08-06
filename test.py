from transformers import DistilBertModel

model = DistilBertModel.from_pretrained("distilbert-base-uncased")

print(len(model.transformer.layer))