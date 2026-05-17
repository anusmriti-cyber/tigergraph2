import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LLMBaselinePipeline:
    def __init__(self, model_id=MODEL_ID, token=HF_TOKEN):
        """
        Initializes the LLM-Only Baseline Pipeline.
        - Loads the trained BERT classifier from train.py to route the query
        - Relies purely on the internal parametric knowledge of the LLM for answering
        """
        self.client = InferenceClient(model=model_id, token=token)
        
        # Load the trained classifier
        self.model_dir = os.path.join(BASE_DIR, 'backend', 'training', 'saved_model')
        if os.path.exists(self.model_dir):
            print(f"Loading trained BERT classifier from {self.model_dir}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.eval()
        else:
            print("Warning: Trained BERT classifier not found. Proceeding without classification.")
            self.tokenizer = None
            self.model = None

    def classify_query(self, query: str) -> str:
        """Classify the query using the BERT model trained in train.py."""
        if not self.model or not self.tokenizer:
            return "Other"
        
        print("Classifying query using trained BERT model...")
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_idx = np.argmax(logits.numpy(), axis=-1)[0]
        
        label_map = {0: 'ASD', 1: 'ADHD', 2: 'Dyscalculia', 3: 'Dyslexia', 4: 'Other'}
        return label_map.get(pred_idx, "Other")

    def generate_answer(self, query: str) -> str:
        """
        Direct prompt -> LLM (No retrieval)
        Uses the trained BERT classifier to identify the condition.
        """
        predicted_condition = self.classify_query(query)
        print(f"Query classified as relating to: {predicted_condition}")

        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful biomedical AI assistant.\n"
                    f"Our trained BERT classification model has routed this query to category: {predicted_condition}.\n"
                    "Answer directly and accurately based only on your internal knowledge."
                )
            },
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat_completion(
                messages,
                max_tokens=512,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"

if __name__ == "__main__":
    pipeline = LLMBaselinePipeline()
    
    sample_query = "What are the common early signs of Autism Spectrum Disorder (ASD) in toddlers?"
    
    print(f"Query: {sample_query}")
    print("-" * 50)
    print("Generating answer (LLM-Only Baseline, NO Retrieval)...")
    
    answer = pipeline.generate_answer(sample_query)
    
    print("-" * 50)
    print(f"Answer:\n{answer}")
