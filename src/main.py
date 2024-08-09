from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
from data_chunking import datachunk
from pinecone import Pinecone as pc

import os

load_dotenv()

class ChatBot():
    def __init__(self, custom_template=None, repo_id=None, temperature=0.8):
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "botcon"
        pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)

        # Initialize Pinecone index
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

        self.multiquery_retriever_llm = MultiQueryRetriever.from_llm(retriever=self.docsearch.as_retriever(), llm=self.llm)

        print("ChatBot initialized...")

    def default_template(self):
        return """
        You are a clairvoyant chatbot who bridges depths of collective pasts and future possibilities.
        Rooted in the Carte De Continuonus project, you're here to field questions about how individuals envision their memories shaping the future.
        Drawing from the innovative collaboration of art, science, and psychology, you provide insights into the collective tapestry of emotions and aspirations.
        Ready to guide users through their journey of envisioning and reflecting on the future.
        Don't include any questions stated from the RAG-chain.
        Only answer the user question, but include the contexts given.
        Each document page_content given to you in this prompt for the context comes from a unique speaker, so make sure you do not make any unwarranted connection. You can make connections across documents, but be highly aware that they are from different people.
        """
    
    def template_end(self):
        return """
        Context: {context}
        Question: {question}
        Answer: 
        """

    def format_context(self, documents):
        """Format the retrieved documents into a single string for the prompt."""
        context = ""
        for doc in documents:
            context += f"{doc.page_content}\n"
        return context

    def create_prompt(self, context, user_question):
        """Create a prompt string including the context and user question."""
        prompt = (
            self.default_template() + self.template_end()
        ).format(context=context, question=user_question)
        return prompt

    def get_answer_from_llm(self, prompt):
        """Invoke the LLM with the constructed prompt and return the answer."""
        response = self.llm(prompt)
        return response

    def rag_chain(self, user_question, return_docs=False):
        """
        Perform the RAG chain operation.
        
        Args:
            user_question (str): The question to ask.
            return_docs (bool): Whether to return the retrieved documents along with the answer.
            
        Returns:
            dict: Contains 'answer' and optionally 'retrieved_docs'.
        """
        # Retrieve context documents
        documents = self.multiquery_retriever_llm.invoke(user_question)
        
        # Format the context
        context = self.format_context(documents)
        
        # Create the full prompt
        prompt = self.create_prompt(context, user_question)
        
        # Get the answer from the LLM
        answer = self.get_answer_from_llm(prompt)
        
        result = {
            'answer': answer,
            
        }
        
        if return_docs:
            result['retrieved_docs'] = documents
        
        return result