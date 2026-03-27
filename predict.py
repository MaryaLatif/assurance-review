import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LOCAL_MODEL = "distilbert_review_model"
BASE_MODEL = "distilbert-base-uncased"

def load_model():
    if os.path.exists(LOCAL_MODEL):
        print(f"✅ Chargement du modèle fine-tuné : {LOCAL_MODEL}")
        path = LOCAL_MODEL
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
    else:
        print(f"⚠️ Modèle fine-tuné non trouvé, chargement du modèle de base : {BASE_MODEL}")
        path = BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=5,
            ignore_mismatched_sizes=True
        )
    model.eval()
    return tokenizer, model, os.path.exists(LOCAL_MODEL)

def predict_review(text, tokenizer, model, device="cpu"):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = probs.argmax(dim=-1).item() + 1

    probs_np = probs.cpu().numpy()[0]
    labels = [f"{i}⭐" for i in range(1, 6)]

    return {
        "prediction": pred,
        "confidence": float(probs_np[pred - 1]),
        "probabilities": {label: float(p) for label, p in zip(labels, probs_np)}
    }
