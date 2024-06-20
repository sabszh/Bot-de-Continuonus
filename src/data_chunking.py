from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
from pinecone import Pinecone as pc
from pinecone import PodSpec
import os
import sys
sys.path.append("..")  

load_dotenv()

def datachunk():
    json_file = os.path.join("data","all.json")
    documents = []

    # Define the metadata extraction function.
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["sender_name"] = record.get("name", "")
        metadata["timestamp_ms"] = record.get("date", "")
        metadata["location"] = record.get("location","")
        metadata["slug"] = record.get("slug","")
        
        emotions = [point["emotion"] for point in record.get("points", [])]
        metadata["emotions"] = emotions
        
        return metadata

    loader = JSONLoader(
        file_path=json_file,
        jq_schema='.[]',
        content_key="text",
        metadata_func=metadata_func
    )

    documents.extend(loader.load())
    
    return documents