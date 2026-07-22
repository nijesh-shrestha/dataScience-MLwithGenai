"""
pinecone --> client

gemini --> client --> embedding --> gemini-embedding-001

pdf --> text extract
extracted text --> embedding --> gemini-embedding-001
embedded text --> pinecone --> vector store

metadata --> pinecone --> vector store
"""

from dotenv import load_dotenv
import os
from pinecone import Pinecone
from google import genai
import fitz 

load_dotenv()

pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
vector_index = pinecone_client.Index("nijesh")

google_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def embed_text(text):
    """
    Embed text using the Gemini embedding model.
    """
    response = google_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "output_dimensionality": 768
        }
    )
    vector = response.embeddings[0].values
    return vector


def upsert_vectors_to_pinecone(document_texts):
    upsert_data = []
    for doc_id, text in enumerate(document_texts):
        vector = embed_text(text)
        record_id = f"doc_{doc_id}"
        metadata = {
            "text": text,
        }
        upsert_data.append((record_id, vector, metadata))
    
    vector_index.upsert(vectors=upsert_data)


if __name__ == "__main__":
    pdf_dir = "docs"
    document_dirs = os.listdir(pdf_dir)

    document_texts = []
    for document_dir in document_dirs:
        document_path = os.path.join(pdf_dir, document_dir)
        text = extract_text_from_pdf(document_path)
        document_texts.append(text)

    upsert_vectors_to_pinecone(document_texts)
    print("Vectors upserted to Pinecone successfully.")