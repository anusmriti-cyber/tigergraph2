import os
import glob
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from transformers import BertTokenizer, BertForSequenceClassification

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'Dataset', 'papers_txt')
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_store')

class BasicRAGPipeline:
    def __init__(self, embed_model_id='all-MiniLM-L6-v2', llm_model_id=MODEL_ID, token=HF_TOKEN):
        """
        Initializes the Basic RAG Pipeline.
        - Chunk papers
        - Embed with sentence-transformers
        - Store in Faiss vector DB
        - Loads the trained BERT classifier from train.py to route the query
        - Retrieve and answer via LLM
        """
        print("Loading Embedding Model...")
        self.embedder = SentenceTransformer(embed_model_id)
        self.llm_client = InferenceClient(model=llm_model_id, token=token)
        self.index = None
        self.chunks = []
        
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
        
    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Simple fixed-size chunking with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def build_or_load_index(self):
        """Builds a new FAISS index from papers or loads an existing one."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
        chunks_path = os.path.join(INDEX_DIR, "chunks.npy")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(index_path)
            self.chunks = np.load(chunks_path, allow_pickle=True).tolist()
            print(f"Loaded {len(self.chunks)} chunks from vector store.")
        else:
            print(f"Building new FAISS index from papers in {DATA_DIR}...")
            texts = []
            for filepath in glob.glob(os.path.join(DATA_DIR, '*.txt')):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
            
            print(f"Loaded {len(texts)} papers. Chunking...")
            for text in texts:
                self.chunks.extend(self._chunk_text(text))
            
            print(f"Created {len(self.chunks)} chunks. Embedding (this may take a minute)...")
            embeddings = self.embedder.encode(self.chunks, show_progress_bar=True, convert_to_numpy=True)
            
            # Normalize for cosine similarity using Inner Product
            print("Normalizing vectors and building FAISS index...")
            faiss.normalize_L2(embeddings)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension) 
            self.index.add(embeddings)

            # Save to disk
            print("Saving index to disk...")
            faiss.write_index(self.index, index_path)
            np.save(chunks_path, np.array(self.chunks, dtype=object))
            print("Index built and saved successfully.")

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

    def retrieve(self, query: str, top_k=3):
        """Retrieve the top_k most relevant chunks for the query."""
        if not self.index:
            self.build_or_load_index()
        
        # Embed and normalize the query
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_emb, top_k)
        
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        return retrieved_chunks

    def generate_answer(self, query: str) -> str:
        """Retrieve context and generate an answer using the LLM."""
        predicted_condition = self.classify_query(query)
        print(f"Query classified as relating to: {predicted_condition}")

        print("Retrieving context...")
        context_chunks = self.retrieve(query)
        
        # Combine retrieved chunks into a single context string
        context = "\n\n---\n\n".join(context_chunks)
        
        prompt = (
            "You are a helpful biomedical AI assistant.\n"
            f"Our trained BERT classification model has routed this query to category: {predicted_condition}.\n"
            "Answer the question based ONLY on the provided context.\n"
            "If the context does not contain the answer, say 'I cannot answer based on the provided context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful biomedical AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            print("Generating response from LLM...")
            response = self.llm_client.chat_completion(
                messages,
                max_tokens=512,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"

if __name__ == "__main__":
    # Test the RAG pipeline
    pipeline = BasicRAGPipeline()
    
    # Initialize / Load vector database
    pipeline.build_or_load_index()
    
    sample_query = "What are the common early signs of Autism Spectrum Disorder (ASD) in toddlers?"
    print(f"\nQuery: {sample_query}")
    print("-" * 50)
    
    answer = pipeline.generate_answer(sample_query)
    
    print("-" * 50)
    print(f"Answer:\n{answer}")
