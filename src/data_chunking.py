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

os

load_dotenv()

def datachunk():
    json_file = os.path.join("data","all.json")

    documents = []
    
    loader = JSONLoader(json_file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len)
    docs = text_splitter.split_documents(documents)
    
    return docs