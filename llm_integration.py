from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import boto3
import nltk
import numpy as np
import json
from typing import List, Dict, Any

nltk.download('punkt')


class ClinicalLLMIntegration:
    def __init__(self, aws_region: str = "us-east-1", llama_model_path: str = "./models/llama-2-7b-embeddings.gguf",
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.llama_model = Llama(model_path=llama_model_path, n_ctx=512)  # For generative responses
        self.embed_model = SentenceTransformer(embed_model_name)  # For query embeddings
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.bucket_name = "clinical-documents-bucket"
        self.clinical_terms = {"HbA1c", "adverse event", "primary endpoint", "secondary endpoint", "immunotherapy",
                               "Phase III"}
        self.max_tokens = 512
        self.context_window = []

    def preprocess_query(self, query: str) -> str:
        for term in self.clinical_terms:
            if term.lower() in query.lower():
                query = query.replace(term, f"<TERM>{term}</TERM>")
        return query

    def engineer_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n".join([chunk["text"] for chunk in context[:3]])
        return f"""
        You are a clinical research assistant. Use the context to answer the query.
        **Context**:
        {context_text}
        **Query**:
        {query}
        **Instructions**:
        - Provide JSON response with 'answer', 'confidence', 'source_sections'.
        - Focus on clinical terms: {', '.join(self.clinical_terms)}.
        """

    def process_clinical_query(self, query: str, context: List[Dict], session_id: str) -> Dict:
        processed_query = self.preprocess_query(query)
        self.context_window.append({"query": query, "context": context})
        if len(self.context_window) > 5:
            self.context_window.pop(0)
        prompt = self.engineer_prompt(query, context)
        try:
            # Generate response using LLaMA
            response = self.llama_model(prompt, max_tokens=self.max_tokens, stop=["</s>"], echo=False)
            answer = response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating LLaMA response: {str(e)}")
            answer = f"Processed query: {query}\nContext: {context[0]['text'][:100]}..." if context else "No data found."

        result = {
            "answer": answer,
            "confidence": 0.8 if context else 0.5,
            "source_sections": [chunk["metadata"].get("section", "unknown") for chunk in context]
        }
        structured_output = {"session_id": session_id, "query": query, "response": result,
                             "context": [chunk["text"] for chunk in context]}
        self.s3_client.put_object(Bucket=self.bucket_name, Key=f"metadata/{session_id}_metadata.json",
                                  Body=json.dumps(structured_output))
        return structured_output

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding using SentenceTransformer for FAISS compatibility."""
        processed_query = self.preprocess_query(query)
        try:
            return self.embed_model.encode(processed_query, convert_to_numpy=True).astype('float32')
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return np.zeros(384, dtype='float32')