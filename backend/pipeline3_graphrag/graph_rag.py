import os
import glob
import re
import networkx as nx
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import InferenceClient
from collections import defaultdict

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'Dataset', 'papers_txt')
GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_store')

# Predefined vocabularies for fast heuristic entity extraction (Zero API Latency)
CONDITIONS_MAP = {
    'ASD': ['asd', 'autism', 'autistic', 'asperger'],
    'ADHD': ['adhd', 'attention deficit', 'hyperactivity'],
    'Dyscalculia': ['dyscalculia', 'math', 'calculia'],
    'Dyslexia': ['dyslexia', 'reading', 'lexia']
}

METHODS_VOCAB = ['cbt', 'fmri', 'eeg', 'therapy', 'intervention', 'machine learning', 'deep learning', 'neural network', 'behavioral', 'cognitive']
DATASETS_VOCAB = ['abide', 'adni', 'cohort', 'registry', 'survey', 'dataset', 'clinical trials']
FINDINGS_VOCAB = ['delayed language', 'social communication', 'repetitive behaviors', 'eye contact', 'sensory sensitivities', 'executive function', 'inattention', 'impulsivity', 'reading difficulty', 'number sense']

class GraphRAGPipeline:
    def __init__(self, llm_model_id=MODEL_ID, token=HF_TOKEN):
        """
        Initializes the GraphRAG Pipeline.
        - High Accuracy, Lowest Latency, Least Tokens.
        - Uses NetworkX for in-memory graph multi-hop traversal.
        """
        self.llm_client = InferenceClient(model=llm_model_id, token=token)
        self.graph = nx.Graph()
        self.graph_path = os.path.join(GRAPH_DIR, 'knowledge_graph.gml')
        
        # Load the trained classifier for Semantic Query Routing
        self.model_dir = os.path.join(BASE_DIR, 'backend', 'training', 'saved_model')
        if os.path.exists(self.model_dir):
            print(f"Loading trained BERT classifier from {self.model_dir}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.eval()
        else:
            print("Warning: Trained BERT classifier not found.")
            self.tokenizer = None
            self.model = None

    def _extract_entities(self, text):
        """Fast heuristic entity extractor. Zero LLM tokens used -> maximum speed."""
        text_lower = text.lower()
        entities = {'Condition': set(), 'Method': set(), 'Dataset': set(), 'Finding': set()}
        
        # Extract Conditions
        for condition, keywords in CONDITIONS_MAP.items():
            if any(kw in text_lower for kw in keywords):
                entities['Condition'].add(condition)
                
        # Extract Methods
        for method in METHODS_VOCAB:
            if method in text_lower:
                entities['Method'].add(method.title())
                
        # Extract Datasets
        for ds in DATASETS_VOCAB:
            if ds in text_lower:
                entities['Dataset'].add(ds.title())
                
        # Extract Findings/Symptoms
        for finding in FINDINGS_VOCAB:
            if finding in text_lower:
                entities['Finding'].add(finding.title())
                
        return entities

    def build_or_load_graph(self):
        """Builds Knowledge Graph from papers or loads it from disk."""
        os.makedirs(GRAPH_DIR, exist_ok=True)

        if os.path.exists(self.graph_path):
            print("Loading existing Knowledge Graph...")
            self.graph = nx.read_gml(self.graph_path)
            print(f"Loaded Graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        else:
            print(f"Building new Knowledge Graph from papers in {DATA_DIR}...")
            txt_files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
            
            for filepath in txt_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    # Extract entities from paper
                    extracted = self._extract_entities(text)
                    
                    # Create nodes and dense edges (Relationships)
                    conditions = list(extracted['Condition'])
                    
                    # Add Condition nodes
                    for cond in conditions:
                        self.graph.add_node(cond, type='Condition')
                        
                        # Link Condition to Methods (Condition -> treated/studied_with -> Method)
                        for method in extracted['Method']:
                            self.graph.add_node(method, type='Method')
                            self.graph.add_edge(cond, method, relation='studied_with')
                            
                        # Link Condition to Datasets
                        for ds in extracted['Dataset']:
                            self.graph.add_node(ds, type='Dataset')
                            self.graph.add_edge(cond, ds, relation='found_in_dataset')
                            
                        # Link Condition to Findings/Symptoms
                        for finding in extracted['Finding']:
                            self.graph.add_node(finding, type='Finding')
                            self.graph.add_edge(cond, finding, relation='exhibits_symptom')
                            
                except Exception as e:
                    pass
            
            print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            print("Saving graph to disk...")
            nx.write_gml(self.graph, self.graph_path)

    def classify_query(self, query: str) -> str:
        """Classify the query using the BERT model trained in train.py."""
        if not self.model or not self.tokenizer:
            return None
        
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_idx = np.argmax(logits.numpy(), axis=-1)[0]
        
        label_map = {0: 'ASD', 1: 'ADHD', 2: 'Dyscalculia', 3: 'Dyslexia', 4: 'Other'}
        condition = label_map.get(pred_idx, "Other")
        return condition if condition != "Other" else None

    def retrieve_multihop_context(self, query: str):
        """
        Multi-hop reasoning: Traverses graph starting from identified entities.
        Returns ultra-compressed token string (Lowest Token Size).
        """
        if self.graph.number_of_nodes() == 0:
            self.build_or_load_graph()
            
        # 1. Semantic routing via trained BERT classifier
        start_node = self.classify_query(query)
        
        # If BERT fails to classify, fallback to keyword extraction from query
        if not start_node or start_node not in self.graph:
            extracted = self._extract_entities(query)
            if extracted['Condition']:
                start_node = list(extracted['Condition'])[0]
            else:
                # Attempt to find any exact matching node
                for node in self.graph.nodes():
                    if str(node).lower() in query.lower():
                        start_node = node
                        break

        if not start_node or start_node not in self.graph:
            return "No relevant entities found in the Knowledge Graph for this query."

        # 2. Multi-hop traversal (up to 2 hops)
        print(f"Graph Traversal starting at node: '{start_node}'")
        subgraph_nodes = nx.single_source_shortest_path_length(self.graph, start_node, cutoff=2).keys()
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # 3. Format into strict, dense context triplets
        context_lines = []
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', 'related_to')
            context_lines.append(f"({u}) -[{relation}]-> ({v})")
            
        compressed_context = "\n".join(context_lines)
        return compressed_context

    def generate_answer(self, query: str) -> str:
        """Retrieve compressed graph context and generate an answer using the LLM."""
        print("Performing Multi-Hop Graph Traversal...")
        context = self.retrieve_multihop_context(query)
        
        print(f"Graph Context Retrieved (Tokens are massively reduced!):\n{context}\n")
        
        prompt = (
            "You are a biomedical AI capable of multi-hop reasoning over a Knowledge Graph.\n"
            "Below is a highly precise subgraph of relationships extracted from our dataset.\n"
            "Answer the question using ONLY the provided graph relationships. Be concise.\n\n"
            f"Knowledge Graph Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        messages = [
            {"role": "system", "content": "You are a specialized GraphRAG biomedical assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            print("Generating response from LLM (Low Latency)...")
            response = self.llm_client.chat_completion(
                messages,
                max_tokens=256, # Short, crisp answer
                temperature=0.1 # High precision
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"

if __name__ == "__main__":
    pipeline = GraphRAGPipeline()
    pipeline.build_or_load_graph()
    
    sample_query = "What are the common early signs of Autism Spectrum Disorder (ASD) in toddlers?"
    print(f"\nQuery: {sample_query}")
    print("-" * 50)
    
    answer = pipeline.generate_answer(sample_query)
    
    print("-" * 50)
    print(f"Answer:\n{answer}")
