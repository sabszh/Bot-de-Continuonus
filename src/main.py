from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
from pinecone import Pinecone as pc
from pinecone import ServerlessSpec
import os

from data_chunking import datachunk

load_dotenv()

class ChatBot():
    
    embeddings = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-large")
    pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=embeddings)
    
    index_name = "futurebot"
    spec = ServerlessSpec(cloud="aws",region="us-east-1")
    
    if index_name not in pinecone_instance.list_indexes().names():
        print("Creating new Pinecone index and loading documents")
        docs = datachunk()
        pinecone_instance.create_index(name=index_name, metric="cosine", dimension=1024, spec=spec)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        print("Created new Pinecone index and loaded documents")
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        print("Using existing Pinecone index")
    
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    
    llm = None  # Define llm, it will be initialized in __init__

    default_template = """
    You are a clairvoyant chatbot who bridges depths of collective pasts and future possibilities.
    Rooted in the Carte De Continuonus project, you're here to field questions about how individuals envision their memories shaping the future.
    Drawing from the innovative collaboration of art, science, and psychology, you provide insights into the collective tapestry of emotions and aspirations.
    Ready to guide users through their journey of envisioning and reflecting on the future.
    """
    
    template_end = """
    Context: {context}
    Question: {question}
    Answer: 
    
    """
    
    def __init__(self, custom_template=None, repo_id=None, temperature=0.8, retriever_method='multiquery_retriever_llm'):
        if custom_template:
            self.template = custom_template
        else:
            self.template = self.default_template
        
        self.temperature = temperature
        self.repo_id = repo_id
        self.retriever_method = retriever_method
        docsearch = self.docsearch
        
        # Initialize llm with repo_id
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            temperature=temperature,
            top_p=0.8,
            top_k=50,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        if retriever_method == "docsearch":
            retriever_method = docsearch.as_retriever()
        elif retriever_method == "multiquery_retriever_llm":
            multiquery_retriever_llm = MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(), llm=self.llm)
            retriever_method = multiquery_retriever_llm


        # Initialize the chain
        self.rag_chain = (
            {"context": retriever_method, "question": RunnablePassthrough()}
            | PromptTemplate(template=self.template+self.template_end, input_variables=["context", "question"])
            | self.llm
            | StrOutputParser()
        )

        print("Chain assembled...")