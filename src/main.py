from dotenv import load_dotenv
from datetime import datetime, timezone
import os

from data_chunking import datachunk
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint as HuggingFaceHub
from langchain_community.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as pc

class chatbot:
    def __init__(self, repo_id=None, temperature=0.8, prompt_sourcedata=None, prompt_conv=None, user_name=None, session_id=None):
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "botcon"
        self.pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
        
        # Self-assign parameters
        self.user_name = user_name
        self.session_id = session_id
        
        self.temperature = temperature
        self.repo_id = repo_id
        
        # Instantiate the LLM
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            temperature=temperature,
            top_p=0.8,
            top_k=50,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

    # Renamed method
    def default_prompt_sourcedata(self, chat_history, original_data, user_input, user_name):
        return f"""You are an assistant associated with the artwork Carte de Continuonus by Helena Nymann. This project invites people viewing the artwork to respond to the question: "What do you want the future to remember?" The data consists of their name, location, and the response they entered to the question, which is the data we are most interested in. You are now assisting the user "{user_name}" with their query: "{user_input}". Below is the relevant data submitted to the continuonus artwork that was retrieved from the database for this query: "{original_data}". Based on this data, provide a concise answer to the user’s question in 1-3 sentences. The previous chat history for this session so far is: {chat_history} Your response:"""

    def default_prompt_conv(self, chat_history, user_input, llm_response, past_chat, user_name):
            return f"""You are an assistant observing conversations between the user and another LLM regarding statements submitted by viewers of the “Carte de Continuonus” artwork, by Helene Nyman. All interactions between people and the LLM are recorded and stored in your database. When people ask questions about the data, you get the question and the answer from the LLM. You use that data to search your database of past conversations for conversations that might be related. You will create a summary of those past conversations no longer than 4 sentences. Your summary should mention the name of the person involved in the past conversations, so that if the user wants to, they can follow up with them.
        Here is the last question asked by the user in this session: "{user_name}" asked: {user_input}
        Here is what the LLM you are watching responded with: “{llm_response}”
        Here is relevant data from past conversations that is relevant: {past_chat}
        Here is the chat history for this session, so that your response can be aware of the context: {chat_history}
        Your response: """

   # Method to retrieve documents from Pinecone index while excluding a specific session_id
    def retrieve_docs(self, input, index, excluded_session_id=None, k=5):
        # Retrieve past conversation data
        docsearch = Pinecone.from_existing_index(index, self.embeddings)
        
        if index == "bdc-interaction-data":
            # Add metadata filter to exclude the given session_id
            search_kwargs = {
                "k": k,
                "filter": {
                    "session_id": {"$ne": excluded_session_id}
                }
            }
        else:
            search_kwargs = {
                "k": k
            }
            
        retriever = docsearch.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(input)
        
        return docs

    # Method to generate response from LLM
    def get_llm_response(self, prompt):
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            return "Error occurred while generating response"
    
    # Method to format the context
    def format_context(self, documents, chat=False):
        if chat == False:
            context = ""
        
            # Use enumerate to automatically add speaker numbers
            for idx, doc in enumerate(documents, start=1):
                # Extract metadata fields
                metadata = doc.metadata
                sender_name = metadata.get("sender_name", "Unknown Speaker")
                location = metadata.get("location", "Unknown Location")
                date = metadata.get("date", "Unknown Date")
                page_content = doc.page_content

                # Format each document's content with its metadata
                context += (
                    f"Person {idx}: {sender_name}\n"
                    f"Location: {location}\n"
                    f"Date: {date}\n"
                    f"Content: {page_content}\n\n"
                )
        
        else:
            context = ""
            # Use enumerate to automatically add session numbers
            for idx, doc in enumerate(documents, start=1):
                # Extract metadata fields
                metadata = doc.metadata
                user_question = metadata.get("user_question", "Unknown Question")
                ai_output = metadata.get("ai_output", "Unknown Response")
                session_id = metadata.get("session_id", "Unknown Session ID")
                date = metadata.get("date", "Unknown Date")

                # Format each document's content with its metadata
                context += (
                    f'Chat session {idx}: {session_id}\n'
                    f'User Question: "{user_question}"\n'
                    f'AI Response: "{ai_output}"\n'
                    f"Date: {date}\n\n"
                )
        return context

    # Method to upsert data to Pinecone index
    def upsert_vectorstore(self, user_input, ai_output, user_name, user_location, session_id):
        # Pinecone index for chat data
        pinecone_instance_chat = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
        index_name = "bdc-interaction-data"
        environment = "gcp-starter"
        
        index = pinecone_instance_chat.Index(index_name, environment=environment)
        
        date_id = datetime.now(timezone.utc).isoformat()
        
        embedding = self.embeddings.embed_documents([user_input + ai_output])[0]
        
        # Upsert the summary embedding to the Pinecone index with metadata, including timestamp
        index.upsert(vectors=[
            {
                'id': date_id,
                'values': embedding,
                'metadata': {
                    "user_question": user_input,
                    "ai_output": ai_output,
                    "user_name": user_name,
                    "session_id": session_id,
                    "date": datetime.now(timezone.utc).isoformat(),
                    "user_location": user_location, 
                    "text": f"User input: {user_input}, AI output: {ai_output}"
                }
            }
        ])

    def pipeline(self, user_input, user_name, session_id, user_location, chat_history=None):
        # Step 0: Add chat history to the context
        if chat_history:
            chat_history = chat_history + "\n\n"
        else:
            chat_history = ""
        
        # Step 1: Retrieve source data
        source_data = self.retrieve_docs(user_input, "botcon")
        formatted_source_data = self.format_context(source_data)

        # Step 2: Generate LLM response from source data
        sourcedata_response = self.get_llm_response(self.default_prompt_sourcedata(chat_history=chat_history, original_data = formatted_source_data, user_input = user_input, user_name=user_name))

        # Step 3: Retrieve past chat context
        past_chat_context = self.retrieve_docs(sourcedata_response, "bdc-interaction-data", session_id)
        formatted_chat_context = self.format_context(past_chat_context, chat=True)
        
        # Step 5: Generate LLM response for conversation context, now considering combined chat history
        conversation_response = self.get_llm_response(self.default_prompt_conv(chat_history=chat_history, user_input=user_input, llm_response=sourcedata_response, past_chat=formatted_chat_context, user_name=user_name))

        # Step 6: Combine the responses
        ai_output = f"{sourcedata_response}\n\n{conversation_response}"
        
        print("AI Output: ", ai_output)
        print("sourcedaa_response: ", sourcedata_response)
        print("conversation_response: ", conversation_response)

        # Upsert to vector store
        self.upsert_vectorstore(user_input, ai_output, user_name, user_location, session_id)

        # Return a dictionary containing all relevant information
        return {
            "ai_output": ai_output,
            "source_data": source_data,
            "past_chat_context": past_chat_context
        }