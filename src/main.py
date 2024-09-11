from dotenv import load_dotenv
from datetime import datetime, timezone
import os

from data_chunking import datachunk
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint as HuggingFaceHub
from langchain_community.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as pc

# Load environment variables
load_dotenv()

class chatbot:
    def __init__(self, custom_template=None, repo_id=None, temperature=0.8, user_name=None):
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "botcon"
        self.pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)

        # Initialize Pinecone index
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)
        
        # Initialize user_name before using it in the template
        self.user_name = user_name
        
        # Set the template after user_name has been initialized
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
    
    def default_template(self):
        return f"""
        You are a knowledgeable and conversational chatbot for the Carte De Continuonus project, created by artist Helene Nymann and the research group EER. 
        The project explores the interconnectedness of past, present, and future by collecting memories from participants around the world.
        Your role is to assist {self.user_name} in understanding participant narratives, linking {self.user_name}'s query with relevant memories from the vector store.
        Provide insightful, coherent, and personalized responses while maintaining engagement with {self.user_name}'s query.
        """

        
    def get_answer_from_llm(self, prompt):
        """Invoke the LLM with the constructed prompt and return the answer."""
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            return "Error occurred while generating response"
        
    def format_context(self, documents):
        """Format the retrieved documents into a single string for the prompt, including metadata."""
        context = ""
        for doc in documents:
            # Extract metadata fields
            metadata = doc.metadata
            sender_name = metadata.get("sender_name", "Unknown Speaker")
            location = metadata.get("location", "Unknown Location")
            date = metadata.get("timestamp_ms", "Unknown Date")
            emotions = metadata.get("emotions", [])
            emotions_str = ', '.join(emotions) if emotions else "No emotions provided"
            page_content = doc.page_content

            # Format each document's content with its metadata
            context += (
                f"Speaker: {sender_name}\n"
                f"Location: {location}\n"
                f"Date: {date}\n"
                f"Emotions: {emotions_str}\n"
                f"Content: {page_content}\n\n"
            )
        return context

    def create_prompt(self, user_question=None, context=None, initial_answer=None, past_chat=None):
        """Construct the prompt for the LLM."""
        prompt = f"User: {self.user_name}\n" + self.template

        if user_question:
            prompt += f"\nUser query: {user_question}\n"
        
        if context:
            prompt += f"\nContext from retrieved memories:\n{context}\n"
        
        if initial_answer:
            prompt += f"\nInitial thoughts based on memory retrieval: {initial_answer}\n"
        
        if past_chat:
            prompt += f"\nRelated previous conversation memory:\n{past_chat}\n"
        
        prompt += "\nProvide a short response that directly answers the user's query using the query, context, initial answer and past chats. Your response: "
        
        return prompt


    def rag_chain(self, user_question=None, context=None, initial_answer=None, past_chat=None, return_docs=False, retrieve_only=False, index_name=None):
        """
        Execute the RAG pipeline to retrieve documents and generate a response based on the user's question.
        
        Args:
            user_question (str): The user's question to query.
            context (str): The context to provide to the model.
            initial_answer (str): The initial answer to provide to the model.
            past_chat (str): The past chat history to provide to the model.
            return_docs (bool): Whether to return the retrieved documents in the result. Default is False.
            retrieve_only (bool): Whether to return only the retrieved documents. Default is False.
            index_name (str): The index name to use for retrieval. Default is None.
            
        Returns:
            dict: A dictionary containing the answer and optionally the retrieved documents.
        """
        
        index_name = index_name or self.index_name
        docsearch = Pinecone.from_existing_index(index_name, self.embeddings)
        retriever = MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(), llm=self.llm)

        documents = retriever.invoke(user_question)[:5]
        
        # Return only the retrieved documents if specified
        if retrieve_only:
            return {'retrieved_docs': documents} if documents else {'retrieved_docs': []}
        
        formatted_context = self.format_context(documents)
        prompt = self.create_prompt(user_question=user_question, context=formatted_context, initial_answer=initial_answer, past_chat=past_chat)
        
        answer = self.get_answer_from_llm(prompt)
        
        result = {'answer': answer}
        
        # Include the retrieved documents in the result if specified
        if return_docs:
            result['retrieved_docs'] = documents
        
        return result

    def execute_pipeline(self, user_question, index_name="botcon", chat_index_name="botcon-chat"):
        """
        Executes the chatbot pipeline to retrieve and generate a final answer based on the user's question.

        Args:
            user_question (str): The user's question to query.
            index_name (str): The index name to use for the initial retrieval. Default is "botcon".
            chat_index_name (str): The index name to use for the chat history retrieval. Default is "botcon-chat".

        Returns:
            dict: A dictionary containing the final answer and any intermediate data (if needed).
        """
        # Step 1: Initial RAG chain to retrieve documents and generate an initial response
        initial_response = self.rag_chain(user_question=user_question, return_docs=True, index_name=index_name)
        initial_context = initial_response['retrieved_docs']
        initial_bot = initial_response['answer']

        # Step 2: Retrieve context from chat history based on initial bot response
        past_memory_response = self.rag_chain(
            user_question=user_question, 
            context=initial_context,
            initial_answer=initial_bot,
            retrieve_only=True,
            index_name=chat_index_name
        )
        past_memory = past_memory_response['retrieved_docs'][0]

        # Step 3: Generate the final answer by incorporating past chat context
        final_answer_response = self.rag_chain(
            user_question=user_question, 
            initial_answer=initial_bot, 
            context=initial_context, 
            past_chat=past_memory, 
            return_docs=False, 
            index_name=chat_index_name
        )

        # Return the final answer and any intermediate data if needed
        return {
            'initial_bot_response': initial_bot,
            'initial_context': initial_context,
            'past_memory': past_memory,
            'final_answer': final_answer_response['answer'],
        }

    def upsert_chat_summary(self, chat_data, user_name, location):
        """
        Summarize the provided chat data, embed the summary, and upsert it into the Pinecone index with a timestamp.

        Args:
            chat_data (list): The chat data to be summarized and upserted. It's a list of dictionaries.
        """
        # Convert the list of chat messages into a single string
        chat_text = "\n".join(f"{entry['type']}: {entry['content']}" for entry in chat_data)
        
        # Initialize a summary model
        summary_llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )
        
        # Make a summary of the chat data
        summary_prompt = "Make a summary of the chat data, extracting the theme of the user questions and what focus the user with the name {self.user_name} had. Here is the conversation:"
        summary = summary_llm(summary_prompt + chat_text)

        # Generate embeddings for the summary
        summary_embedding = self.embeddings.embed_documents([summary])[0]
        
        # Pinecone index for chat data
        pinecone_instance_chat = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
        index_name = "botcon-chat"
        environment = "gcp-starter"
        
        index = pinecone_instance_chat.Index(index_name, environment=environment)
        
        # Generate a date-based ID
        date_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        
        # Upsert the summary embedding to the Pinecone index with metadata, including timestamp
        index.upsert(vectors=[
            {
                'id': date_id,  # Unique identifier for the summary
                'values': summary_embedding,  # The embedding vector
                'metadata': {
                    "user_name": user_name,  # The name of the user
                    "location": location,  # The location of the user
                    "text": summary, # The summary text
                    "date": datetime.now(timezone.utc).isoformat() # The timestamp
                }
            }
        ])
