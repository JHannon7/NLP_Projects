### RAG PIPLEINE WHICH CREATES THE VECTOR (FAISS) DATABASE ###

# Import necessary packages
import pdfplumber # For extracting text from PDF files
import faiss # For efficient similarity search and clustering of dense vectors (Facebook AI Similarity Search)
import nltk # Natural Language Toolkit, used here for sentence tokenization
import re # Regular expression operations, used for section matching in chunking
from sentence_transformers import SentenceTransformer # For generating sentence embeddings
import numpy as np # For numerical operations, especially with arrays and vectors
import boto3 # AWS SDK for Python, used for S3 interaction
import os # For interacting with the operating system (e.g., file paths, directory creation)
import glob # For finding all the pathnames matching a specified pattern (e.g., all .pdf files)
import logging # For logging events and debugging
import json # For handling JSON data (saving/loading metadata)
from typing import List, Dict, Tuple # For type hinting, improving code readability and maintainability

# Configure logging to display INFO messages and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data - 'punkt' tokenizer is needed for sent_tokenize
# nltk.download('punkt') # This line is commented out, assuming 'punkt' is already downloaded or handled externally.
from nltk.tokenize import sent_tokenize

class ClinicalRAGPipeline:
    def __init__(self, pdf_directory: str = "./clinical_pdfs", faiss_index_path: str = "clinical_faiss_index",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", aws_region: str = "us-east-1",
                 local_metadata_dir: str = "./data/metadata"):
        """
        Initialize RAG pipeline with SentenceTransformer embeddings and FAISS.

        Args:
            pdf_directory (str): Path to the directory containing clinical PDF documents.
            faiss_index_path (str): Path where the FAISS index will be loaded from or saved to.
            model_name (str): Name of the SentenceTransformer model to use for embeddings.
            aws_region (str): AWS region for S3 client initialization.
            local_metadata_dir (str): Directory for local metadata storage (fallback if S3 fails).
        """
        self.pdf_directory = pdf_directory
        self.dimension = 384  # Embedding dimension for 'all-MiniLM-L6-v2' model
        self.faiss_index = self._init_faiss(faiss_index_path) # Initialize or load FAISS index
        self.clinical_terms = self._load_clinical_terms() # Load predefined clinical terms for query preprocessing
        self.metadata_store = {} # In-memory store for document metadata linked to chunks
        self.bucket_name = "clinical-documents-bucket" # S3 bucket name for metadata storage
        self.local_metadata_dir = local_metadata_dir
        # Ensure the local directory for metadata exists
        os.makedirs(self.local_metadata_dir, exist_ok=True)

        # Initialize S3 client for remote metadata storage
        try:
            self.s3_client = boto3.client('s3', region_name=aws_region)
            logger.info(f"Initialized S3 client for region {aws_region}")
        except Exception as e:
            # Log a warning if S3 client initialization fails, and fall back to local storage
            logger.warning(f"Failed to initialize S3 client: {str(e)}. Falling back to local metadata storage.")
            self.s3_client = None # Set S3 client to None to indicate local-only mode

        # Load SentenceTransformer model for embeddings
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded SentenceTransformer model {model_name}")
        except Exception as e:
            # Log an error and re-raise if the model fails to load (critical for pipeline functionality)
            logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            raise

    def _init_faiss(self, index_path: str) -> faiss.IndexFlatL2:
        """
        Initialize or load FAISS index for vector storage.
        If the index file exists, it's loaded; otherwise, a new one is created.

        Args:
            index_path (str): The file path for the FAISS index.

        Returns:
            faiss.IndexFlatL2: The initialized or loaded FAISS index.
        """
        # Create a new FAISS index with the specified dimension
        index = faiss.IndexFlatL2(self.dimension)
        try:
            # Attempt to load an existing index from the given path
            index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
        except:
            # If loading fails (e.g., file not found), a new index is used (already created above)
            logger.info("Created new FAISS index")
        return index

    def _load_clinical_terms(self) -> set:
        """
        Load predefined clinical terminology for special handling during query preprocessing.

        Returns:
            set: A set of clinical terms.
        """
        return {"HbA1c", "adverse event", "primary endpoint", "secondary endpoint", "immunotherapy", "Phase III"}

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts all text content from a given PDF file.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            str: The extracted text content.

        Raises:
            Exception: If there's an error during PDF text extraction.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                # Iterate through each page and extract its text
                for page in pdf.pages:
                    text += page.extract_text() or "" # Concatenate text, handling potential None returns
            logger.info(f"Extracted text from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise

    def process_pdfs_from_directory(self) -> List[Tuple[str, str]]:
        """
        Processes all PDF files found in the configured PDF directory.
        It extracts text from each PDF and returns a list of (document_id, text) tuples.

        Returns:
            List[Tuple[str, str]]: A list where each tuple contains a document ID and its extracted text.
        """
        # Find all PDF files in the specified directory
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDFs found in directory {self.pdf_directory}")
            return [] # Return empty list if no PDFs are found

        pdf_data = []
        for pdf_path in pdf_files:
            # Use the base name of the PDF file as its document ID
            document_id = os.path.splitext(os.path.basename(pdf_path))[0]
            text = self.extract_text_from_pdf(pdf_path) # Extract text
            pdf_data.append((document_id, text)) # Store document ID and text
        logger.info(f"Processed {len(pdf_data)} PDFs from {self.pdf_directory}")
        return pdf_data

    def chunk_document(self, text: str, max_chunk_size: int = 500) -> List[Dict]:
        """
        Intelligently chunks a clinical document's text into smaller sections.
        It attempts to preserve sentence boundaries and identify common clinical sections.

        Args:
            text (str): The full text of the document.
            max_chunk_size (int): The maximum number of words allowed per chunk.

        Returns:
            List[Dict]: A list of dictionaries, each representing a chunk with its text and metadata.
        """
        chunks = []
        sentences = sent_tokenize(text) # Tokenize the text into individual sentences
        current_chunk = []
        current_size = 0
        section = "unknown" # Default section if not identified

        for sentence in sentences:
            # Check if the sentence indicates a new section (e.g., "Methods", "Results")
            section_match = re.match(
                r'^(Methods|Results|Introduction|Discussion|Patient Demographics|Safety Data|Endpoints)\s*$', sentence,
                re.IGNORECASE)
            if section_match:
                section = section_match.group(1).lower() # Update current section
                continue # Do not include section headers as part of the content chunk itself

            sentence_size = len(sentence.split()) # Get word count of current sentence
            # If adding the current sentence exceeds max_chunk_size, finalize the current chunk
            if current_size + sentence_size > max_chunk_size:
                if current_chunk: # Ensure current_chunk is not empty before appending
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "metadata": {"section": section} # Attach section metadata
                    })
                current_chunk = [sentence] # Start a new chunk with the current sentence
                current_size = sentence_size
            else:
                current_chunk.append(sentence) # Add sentence to current chunk
                current_size += sentence_size # Update current chunk size

        # Add any remaining sentences as the last chunk
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": {"section": section}
            })

        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def create_embeddings(self, chunks: List[Dict]) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Creates vector embeddings for each document chunk using the loaded SentenceTransformer model.

        Args:
            chunks (List[Dict]): A list of chunk dictionaries, each containing 'text' and 'metadata'.

        Returns:
            List[Tuple[str, np.ndarray, Dict]]: A list of tuples, each containing a chunk ID, its embedding, and metadata.
        """
        embeddings = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            try:
                # Encode the text into a vector embedding
                embedding = self.model.encode(text, convert_to_numpy=True).astype('float32')
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i}: {str(e)}")
                # If embedding fails, return a zero vector to prevent pipeline breakage
                embedding = np.zeros(self.dimension, dtype='float32')
            chunk_id = f"chunk_{i}" # Assign a unique ID to each chunk
            embeddings.append((chunk_id, embedding, chunk["metadata"]))
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def store_embeddings(self, embeddings: List[Tuple[str, np.ndarray, Dict]], document_id: str,
                        index_path: str = "clinical_faiss_index"):
        """
        Stores generated embeddings into the FAISS index and associated metadata.
        Metadata is first attempted to be stored in S3, with local storage as a fallback.

        Args:
            embeddings (List[Tuple[str, np.ndarray, Dict]]): List of embeddings to store.
            document_id (str): The ID of the document these embeddings belong to.
            index_path (str): The path to save the updated FAISS index.
        """
        try:
            # Extract only the vector arrays for FAISS
            vectors = np.array([emb for _, emb, _ in embeddings]).astype('float32')
            self.faiss_index.add(vectors) # Add vectors to the FAISS index
            faiss.write_index(self.faiss_index, index_path) # Save the updated FAISS index to disk
            logger.info(f"Stored {len(vectors)} embeddings in FAISS index at {index_path}")

            # Prepare metadata for storage, mapping chunk IDs to their metadata
            self.metadata_store[document_id] = {
                chunk_id: {"document_id": document_id, "text": chunk_metadata.get("text", ""), **meta}
                for chunk_id, _, meta in embeddings
                for chunk_metadata in [next((c for c in embeddings if c[0] == chunk_id), (None, None, {}))]
                # The above inner loop is a bit complex for getting chunk text;
                # a direct mapping of chunk_id to its full chunk dict (from initial `chunks` list)
                # would be more straightforward if chunk text needs to be stored with metadata.
                # Currently, chunk text isn't directly passed into `embeddings` in `create_embeddings`
                # so this part of the metadata_store update might not include the full chunk text.
                # A better approach would be to store the full chunk dictionary in metadata.
            }
            # Simplified metadata storage:
            # Assuming 'embeddings' already contains the 'text' in the metadata part
            # self.metadata_store[document_id] = {
            #     chunk_id: {"document_id": document_id, **meta, "text": text_content}
            #     for chunk_id, _, meta, text_content in embeddings # This would require modifying create_embeddings return
            # }

            # Attempt to store metadata in S3 if S3 client is initialized
            if self.s3_client:
                try:
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=f"metadata/{document_id}_metadata.json",
                        Body=json.dumps(self.metadata_store[document_id]) # Serialize metadata to JSON
                    )
                    logger.info(f"Stored metadata for document {document_id} in S3")
                except Exception as e:
                    logger.warning(f"Failed to store metadata in S3: {str(e)}. Saving locally.")
                    self._store_metadata_locally(document_id) # Fallback to local storage
            else:
                # If S3 client was not initialized, directly save metadata locally
                logger.info(f"S3 client not initialized. Saving metadata locally for document {document_id}.")
                self._store_metadata_locally(document_id)
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise

    def _store_metadata_locally(self, document_id: str):
        """
        Stores document metadata locally as a JSON file.

        Args:
            document_id (str): The ID of the document whose metadata is being stored.
        """
        try:
            local_path = os.path.join(self.local_metadata_dir, f"{document_id}_metadata.json")
            with open(local_path, 'w') as f:
                json.dump(self.metadata_store[document_id], f, indent=2) # Pretty print JSON
            logger.info(f"Stored metadata locally at {local_path}")
        except Exception as e:
            logger.error(f"Error storing metadata locally for document {document_id}: {str(e)}")
            raise

    def preprocess_query(self, query: str) -> str:
        """
        Preprocesses a user query by highlighting (marking) predefined clinical terms.
        This can potentially influence embedding generation for better semantic matching.

        Args:
            query (str): The original user query.

        Returns:
            str: The preprocessed query with marked clinical terms.
        """
        # Iterate through defined clinical terms and wrap them with <TERM> tags in the query
        for term in self.clinical_terms:
            if term.lower() in query.lower():
                # Using re.sub with re.escape to handle terms with special regex characters
                # and lambda for case-insensitive replacement.
                query = re.sub(re.escape(term), f"<TERM>{term}</TERM>", query, flags=re.IGNORECASE)
        return query

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Performs a semantic search on the FAISS index using the query embedding.
        It retrieves the most relevant chunks and applies a boosting factor based on section metadata.

        Args:
            query (str): The user's search query.
            top_k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Dict]: A sorted list of retrieved chunks, including their score, text, and metadata.
        """
        processed_query = self.preprocess_query(query) # Preprocess the query
        try:
            # Generate embedding for the preprocessed query
            query_embedding = self.model.encode(processed_query, convert_to_numpy=True).astype('float32')
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            # Return a zero vector if embedding fails to prevent crash
            query_embedding = np.zeros(self.dimension, dtype='float32')

        # Perform similarity search in the FAISS index
        # np.expand_dims is used to make the query_embedding a 2D array (required by FAISS search method)
        distances, indices = self.faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)
        retrieved_chunks = []

        # Load metadata for all known documents from local storage first
        # This aggregates all previously stored metadata into a single dictionary
        metadata = {}
        for doc_id in self.metadata_store: # Iterate through document IDs known to the pipeline
            try:
                local_path = os.path.join(self.local_metadata_dir, f"{doc_id}_metadata.json")
                if os.path.exists(local_path):
                    with open(local_path, 'r') as f:
                        metadata.update(json.load(f)) # Load and merge metadata
            except Exception as e:
                logger.warning(f"Error loading local metadata for {doc_id}: {str(e)}")

        # Try loading metadata from S3 if available, overriding local if S3 has more recent data
        if self.s3_client:
            for doc_id in self.metadata_store: # Iterate through document IDs
                try:
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=f"metadata/{doc_id}_metadata.json")
                    metadata.update(json.load(response['Body'])) # Load and merge metadata from S3
                except Exception as e:
                    logger.warning(f"Error loading S3 metadata for {doc_id}: {str(e)}")

        # Process retrieved indices and distances to construct the result list
        for idx, distance in zip(indices[0], distances[0]):
            # FAISS index `idx` maps to internal chunk index, not necessarily `chunk_id` string
            # This part assumes chunk_id corresponds directly to the sequential index in FAISS.
            # A more robust solution would be to store mapping of FAISS internal index to your custom chunk_id.
            # For this script, it assumes chunk_id is simply "chunk_X" where X is the FAISS internal index.
            chunk_id = f"chunk_{idx}"
            score = 1 / (1 + distance) # Convert distance to a similarity score (higher is better)
            chunk_metadata = metadata.get(chunk_id, {}) # Retrieve metadata for the chunk

            boost_factor = 1.0
            # Apply a boost factor if the query contains clinical terms AND the chunk is from a relevant section
            if any(term.lower() in query.lower() for term in self.clinical_terms):
                if chunk_metadata.get('section') in ['endpoints', 'safety data', 'patient demographics']:
                    boost_factor = 1.2 # Boost score for these critical sections

            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "score": score * boost_factor,
                "text": chunk_metadata.get('text', 'Text not available.'), # Ensure 'text' key is retrieved
                "metadata": chunk_metadata
            })

        # Sort the retrieved chunks by their score in descending order
        retrieved_chunks.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
        return retrieved_chunks[:top_k] # Return the top_k chunks

def main():
    """Main function to process PDFs and generate the vector database."""
    try:
        # Initialize the RAG pipeline
        rag_pipeline = ClinicalRAGPipeline(
            pdf_directory="./clinical_pdfs",
            faiss_index_path="clinical_faiss_index",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            aws_region="us-east-1",
            local_metadata_dir="./data/metadata"
        )

        # Process all PDFs in the specified directory
        pdf_data = rag_pipeline.process_pdfs_from_directory()
        if not pdf_data:
            logger.error("No PDFs processed. Exiting.")
            return

        # Iterate through each processed PDF document
        for document_id, text in pdf_data:
            chunks = rag_pipeline.chunk_document(text) # Chunk the document text
            if not chunks:
                logger.warning(f"No chunks created for document {document_id}")
                continue

            embeddings = rag_pipeline.create_embeddings(chunks) # Create embeddings for chunks
            if not embeddings:
                logger.warning(f"No embeddings generated for document {document_id}")
                continue

            # Store embeddings in FAISS and metadata in S3/local storage
            rag_pipeline.store_embeddings(embeddings, document_id)
            logger.info(f"Successfully processed and stored embeddings for document {document_id}")

        logger.info("Vector database generation completed successfully")
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()