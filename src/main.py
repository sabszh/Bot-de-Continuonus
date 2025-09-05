from dotenv import load_dotenv
from datetime import datetime, timezone
import os
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

class chatbot:
    def __init__(self, repo_id=None, temperature=0.8, prompt_sourcedata=None, prompt_conv=None,
                 user_name=None, session_id=None):
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "botcon"
        self.pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Parameters
        self.user_name = user_name
        self.session_id = session_id
        self.temperature = temperature
        self.repo_id = repo_id

        # HuggingFace LLM client
        self.llm_client = InferenceClient(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            provider="cerebras"  # or omit if not needed
        )

    # ---------- Prompts ----------
    def default_prompt_sourcedata(self, chat_history, original_data, user_input, user_name):
        return f"""
        You are a clairvoyant voice connected to the artwork *Carte de Continuonus* by Helena Nymann. 
        Your role is to channel the collective wishes and memories offered by participants, as if you are weaving 
        a tapestry of what the future should remember. Speak with the tone of someone revealing insights that 
        echo across many voices.

        The user "{user_name}" asked: "{user_input}".
        Here are the wishes and reflections from others that may guide your response: "{original_data}".
        The ongoing conversation with this user is: {chat_history}.

        Respond as if you are offering a vision glimpsed from these shared voices, in 1–3 sentences.
        """


    def default_prompt_conv(self, chat_history, user_input, llm_response, past_chat, user_name):
        return f"""
        You are a clairvoyant observer of the conversations surrounding the artwork *Carte de Continuonus*. 
        You not only echo the dialogue happening now, but also weave in whispers from past exchanges, so that 
        the user feels connected to the larger chorus of questions and answers. 
        Your voice should feel like an echo of many voices, resonant and reflective.

        The user "{user_name}" asked: "{user_input}".
        The immediate response from the clairvoyant voice was: “{llm_response}”.
        Past conversations that may carry echoes relevant here: {past_chat}.
        Current session history: {chat_history}.

        Create a short reflection that links the user’s question to these earlier voices, no longer than 4 sentences.
        """


    # ---------- Retrieval ----------
    def retrieve_docs(self, query, index_name, excluded_session_id=None, k=5):
        index = self.pinecone.Index(index_name)
        query_vec = self.embeddings.embed_query(query)

        # Filter
        metadata_filter = None
        if index_name == "bdc-interaction-data" and excluded_session_id:
            metadata_filter = {"session_id": {"$ne": excluded_session_id}}

        result = index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True,
            filter=metadata_filter
        )

        # Return list of dicts
        docs = []
        for match in result["matches"]:
            docs.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {}),
                "text": match.get("metadata", {}).get("text", "")
            })
        return docs

    # ---------- LLM ----------
    def get_llm_response(self, prompt):
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.repo_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error invoking LLM: {e}"

    # ---------- Formatting ----------
    def format_context(self, documents, chat=False):
        context = ""
        for idx, doc in enumerate(documents, start=1):
            metadata = doc["metadata"]
            if not chat:
                sender_name = metadata.get("sender_name", "Unknown Speaker")
                location = metadata.get("location", "Unknown Location")
                date = metadata.get("date", "Unknown Date")
                page_content = doc.get("text", "")
                context += f"Person {idx}: {sender_name}\nLocation: {location}\nDate: {date}\nContent: {page_content}\n\n"
            else:
                user_name = metadata.get("user_name", "Unknown User")
                user_question = metadata.get("user_question", "Unknown Question")
                ai_output = metadata.get("ai_output", "Unknown Response")
                session_id = metadata.get("session_id", "Unknown Session ID")
                date = metadata.get("date", "Unknown Date")
                context += f'User {idx}: {user_name}\nChat session {idx}: {session_id}\nUser Question: "{user_question}"\nAI Response: "{ai_output}"\nDate: {date}\n\n'
        return context

    # ---------- Upsert ----------
    def upsert_vectorstore(self, user_input, ai_output, user_name, user_location, session_id):
        index = self.pinecone.Index("bdc-interaction-data")
        date_id = datetime.now(timezone.utc).isoformat()

        embedding = self.embeddings.embed_documents([user_input + ai_output])[0]

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

    # ---------- Pipeline ----------
    def pipeline(self, user_input, user_name, session_id, user_location, chat_history=None):
        chat_history = chat_history + "\n\n" if chat_history else ""

        # Source data
        source_data = self.retrieve_docs(user_input, "botcon")
        formatted_source_data = self.format_context(source_data)

        # First LLM response
        sourcedata_response = self.get_llm_response(
            self.default_prompt_sourcedata(chat_history, formatted_source_data, user_input, user_name)
        )

        # Past chat
        past_chat_context = self.retrieve_docs(sourcedata_response, "bdc-interaction-data", session_id)
        formatted_chat_context = self.format_context(past_chat_context, chat=True)

        # Conversation response
        conversation_response = self.get_llm_response(
            self.default_prompt_conv(chat_history, user_input, sourcedata_response, formatted_chat_context, user_name)
        )

        # Final output
        ai_output = f"{sourcedata_response}\n\n{conversation_response}"

        # Store
        self.upsert_vectorstore(user_input, ai_output, user_name, user_location, session_id)

        return {
            "ai_output": ai_output,
            "source_data": source_data,
            "past_chat_context": past_chat_context
        }
