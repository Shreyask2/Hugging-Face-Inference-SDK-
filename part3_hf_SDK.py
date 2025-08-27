from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
from io import BytesIO
import torch

print("Running NLP text generation...")
nlp_generator = pipeline("text-generation", model="gpt2")
nlp_result = nlp_generator("Write a short poem about freedom", max_new_tokens=500)
nlp_text = nlp_result[0]['generated_text']
print("=== NLP Output ===")
print(nlp_text)

print("\nRunning Vision image classification...")

image_url = "https://wpengine.com/wp-content/uploads/2021/05/optimize-images.jpg"

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(image_url, headers=headers)
response.raise_for_status()  # Check that download succeeded

image = Image.open(BytesIO(response.content)).convert("RGB")

feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vision_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = vision_model(**inputs)
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
pred_label = vision_model.config.id2label[probs.argmax().item()]

print("=== Vision Output ===")
print(pred_label)

with open("inference_results.txt", "w") as f:
    f.write("=== NLP Output ===\n")
    f.write(nlp_text + "\n\n")
    f.write("=== Vision Output ===\n")
    f.write(pred_label + "\n")

print("\nAll results saved to 'inference_results.txt'")
