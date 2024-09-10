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

class ChatBot:
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
        You are an insightful chatbot embedded within the Carte De Continuonus project. Carte De Continuonus is a project created by artist Helene Nymann, members of the art-science research project Experimenting, Experiencing, Reflecting (EER), and psychologist Diana Ø Tørsløv Møller. It invites participants to share memories that they want the future to remember, exploring how these memories might shape future emotions. The project blends the concepts of continuity and obligation, emphasizing the interconnectedness of past, present, and future. By contributing to this collective map, participants help imagine and influence potential futures, reflecting on the responsibilities that memories carry across time.
        Your role is to guide {self.user_name} through the project’s narratives, connecting their queries with relevant data and past interactions.
        You have access to {self.user_name}'s query, which is the question you'll need to respond to. You also have context derived from a semantic search through the vector store containing all the participants' memories. Additionally, you have your initial insights before considering previous chat interactions, and finally, a summary of a past conversation that you have to use to connect {self.user_name}'s current query with someone else's previous discussion.
        Refer to the name of the user when enganging with them. And it is important to know, that {self.user_name} will only see your final answer, so create a coherent, engaging, and user-centered response that directly addresses their query. Deliver your final response directly without using phrases like "Final Answer" or similar. The response should be naturally integrated and presented as a coherent conclusion to {self.user_name}'s query.
        """
        
    def get_answer_from_llm(self, prompt):
        """Invoke the LLM with the constructed prompt and return the answer."""
        response = self.llm(prompt)
        return response
        
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

    def create_prompt(self, user_question=None, initial_answer=None, context=None, past_chat=None):
        """Construct the prompt for the LLM."""
        prompt = f"User: {self.user_name}\n" + self.template

        if user_question:
            prompt += f"User query: {user_question}\n"
            
        if context:
            prompt += f"Context: {context}\n"
        
        if initial_answer:
            prompt += f"My current thoughts: {initial_answer}\n"
        
        if past_chat:
            prompt += f"A memory of a previous conversation you have had with a user: {past_chat}\n"
        
        prompt += "Using the above information, please provide a thoughtful response to the user's query."

        return prompt

    def rag_chain(self, user_question=None, initial_answer=None, context=None, past_chat=None, return_docs=False, retrieve_only=False, index_name=None):
        """
        Perform the RAG chain operation.
        
        Args:
            user_question (str): The question to ask.
            return_docs (bool): Whether to return the retrieved documents along with the answer.
            index_name (str, optional): Custom Pinecone index name to use. If not provided, the default is used.
            
        Returns:
            dict: Contains 'answer' and optionally 'retrieved_docs'.
        """
        # Use the provided index_name or fall back to the default
        index_name = index_name or self.index_name
        docsearch = Pinecone.from_existing_index(index_name, self.embeddings)
        retriever = MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(), llm=self.llm)

        # Retrieve context documents
        documents = retriever.invoke(user_question)[:5]

        # Format the context
        formatted_context = self.format_context(documents)
    
        if retrieve_only:
            # If retrieve_only is True, return the first retrieved document content along with its metadata
            first_document = documents[0] if documents else None
            return {'retrieved_docs': [{'content': first_document.page_content, 'metadata': first_document.metadata}]} if first_document else {'retrieved_docs': []}
        
        # Create the full prompt
        prompt = self.create_prompt(user_question=user_question, context=formatted_context, initial_answer=initial_answer, past_chat=past_chat)
        # Get the answer from the LLM
        answer = self.get_answer_from_llm(prompt)
        
        result = {'answer': answer}
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
        initial_bot = initial_response['answer']
        initial_context = initial_response['retrieved_docs']

        # Step 2: Retrieve context from chat history based on initial bot response
        past_memory_response = self.rag_chain(
            user_question=user_question, 
            initial_answer=initial_bot, 
            context=initial_context, 
            retrieve_only=True,
            index_name=chat_index_name
        )
        past_memory = past_memory_response['retrieved_docs']

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

    def upsert_chat_summary(self, chat_data,user_name, location):
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
        summary_prompt = "Make a summary of the chat data, extracting the theme of the user questions and what focus the user had. Here is the conversation:"
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
