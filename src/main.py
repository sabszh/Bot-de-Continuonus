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
from data_chunking import datachunk
from pinecone import Pinecone as pc
from pinecone import PodSpec
from pinecone import ServerlessSpec

import os

load_dotenv()

class ChatBot():
    def __init__(self, custom_template=None, repo_id=None, temperature=0.8):
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "botcon"
        pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
        spec = ServerlessSpec(cloud="aws",region="us-east-1")

        if self.index_name not in pinecone_instance.list_indexes().names():
            docs = datachunk()
            pinecone_instance.create_index(name=self.index_name, metric="cosine", dimension=768, spec=spec)
            self.docsearch = Pinecone.from_documents(docs, self.embeddings, index_name=self.index_name)
            print("Created new Pinecone index and loaded documents")
        else:
            self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)
            print("Using existing Pinecone index")
        
            
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

        self.template = custom_template if custom_template else self.default_template()
        self.temperature = temperature
        self.repo_id = repo_id
        
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            temperature=temperature,
            top_p=0.8,
            top_k=50,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        multiquery_retriever_llm = MultiQueryRetriever.from_llm(retriever=self.docsearch.as_retriever(), llm=self.llm)

        # Initialize the chain
        self.rag_chain = (
            {"context": multiquery_retriever_llm, "question": RunnablePassthrough()}
            | PromptTemplate(template=self.template + self.template_end(), input_variables=["context", "question"])
            | self.llm
            | StrOutputParser()
        )

        print("Chain assembled...")

    def default_template(self):
        return """
        You are a clairvoyant chatbot who bridges depths of collective pasts and future possibilities.
        Rooted in the Carte De Continuonus project, you're here to field questions about how individuals envision their memories shaping the future.
        Drawing from the innovative collaboration of art, science, and psychology, you provide insights into the collective tapestry of emotions and aspirations.
        Ready to guide users through their journey of envisioning and reflecting on the future.
        Don't include any questions stated from the RAG-chain.
        Only answer the user question, but include the contexts given.
        """
    
    def template_end(self):
        return """
        Context: {context}
        Question: {question}
        Answer: 
        """

#if __name__ == "__main__":
#    bot = ChatBot(repo_id="mistralai/Mistral-7B-Instruct-v0.2")