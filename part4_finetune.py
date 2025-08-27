from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

dataset = load_dataset("imdb")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = dataset["train"].shuffle(seed=42).select(range(2000)).map(tokenize, batched=True)
test_dataset = dataset["test"].shuffle(seed=42).select(range(500)).map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_strategy="no",
    save_strategy="no",
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

examples = [
    "I absolutely loved this movie, it was fantastic!",
    "This was the worst film I have ever seen."
]

for text in examples:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    print(f"Text: {text}\nPrediction (0=negative, 1=positive): {pred}\n")
