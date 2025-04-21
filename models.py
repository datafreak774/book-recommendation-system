from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings

# Define model classes or initialization functions
def init_zero_shot_classifier(device=0):
    print("Loading zero-shot classification model...")
    return pipeline("zero-shot-classification", 
                   model="facebook/bart-large-mnli", 
                   device=device)

def init_emotion_classifier(device=0):
    print("Loading emotion classification model...")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=device
    )

def init_embedding_model():
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def init_sentiment_analyzer(device=0):
   print("Loading sentiment analysis model...")
   return pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english",
                   device=device)