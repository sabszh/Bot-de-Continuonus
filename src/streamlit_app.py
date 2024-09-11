import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from main import chatbot
import streamlit_nested_layout

# Repositories mapping
repositories = {
    "mistralai": "mistralai/Mistral-7B-Instruct-v0.1",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

st.set_page_config(page_title="Bot de Continuonus", layout="wide", initial_sidebar_state='collapsed')

# Initialize session state variables
if "chat_data" not in st.session_state:
    st.session_state.chat_data = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

# Initialize history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = StreamlitChatMessageHistory()

# Function to capture and set the user's name
@st.dialog("User Name and Location", width="small")
def ask_name():
    user_name = st.text_input("Please enter your name:")
    location = st.text_input("Please enter your location:")
    if st.button("Submit"):
        if user_name and location:
            st.session_state.user_name = user_name
            st.session_state.location = location
            st.rerun()

if st.session_state.user_name is None:
    ask_name()

# Sidebar for LLM settings and chat management
with st.sidebar:
    st.title('LLM Settings')
    st.write(
        """Be aware, if you make any changes here, the chatbot will reload and your chat will be lost.
        If you encounter errors, try a different model or check if the model is overloaded. Try again later if needed."""
    )
    
    # Add print statements to check selected repo and temperature
    selected_repo = st.selectbox("Select the Model Repository", list(repositories.keys()))
    temperature = st.slider("Select the Temperature (0-2)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
    
    custom_prompt = st.text_area('Edit System Prompt', 
        f"""
        You are a knowledgeable and conversational chatbot for the Carte De Continuonus project, created by artist Helene Nymann and the research group EER. 
        The project explores the interconnectedness of past, present, and future by collecting memories from participants around the world.
        Your role is to assist {st.session_state.user_name} in understanding participant narratives, linking {st.session_state.user_name}'s query with relevant memories from the vector store.
        Provide insightful, coherent, and personalized responses while maintaining engagement with {st.session_state.user_name}'s query.
        Focus on using specific information from memories and prior conversations to make your responses detailed and relevant.
        """)        

    if st.button("Save and Reset Chat", key="upsert_button"):
        if "bot" in st.session_state:
            with st.spinner("Saving chat..."):
                try:
                    st.session_state.bot.upsert_chat_summary(st.session_state.chat_data, st.session_state.user_name, st.session_state.location)
                    st.success("Chat successfully saved!")
                    st.session_state.chat_data = []
                    st.session_state.history.clear()
                except Exception as e:
                    st.error(f"Error saving chat: {e}")
        else:
            st.warning("Bot not initialized; cannot save chat.")

# Initialize the chatbot instance if required
def initialize_bot():
    repo_id = repositories.get(selected_repo)
    
    try:
        if ("bot" not in st.session_state or
            st.session_state.custom_prompt != custom_prompt or
            st.session_state.selected_repo != selected_repo or
            st.session_state.temperature != temperature):
            
            st.session_state.bot = chatbot(
                custom_template=custom_prompt, 
                repo_id=repo_id, 
                temperature=temperature,
                user_name=st.session_state.user_name,
            )
            st.session_state.custom_prompt = custom_prompt
            st.session_state.selected_repo = selected_repo
            st.session_state.temperature = temperature
            st.session_state.repo_id = repo_id
            
    except Exception as e:
        st.error(f"Error initializing bot: {e}")
        print(f"Error initializing bot: {e}")

initialize_bot()

# Function to generate a response from the chatbot
def generate_response(input_text):
    bot = st.session_state.get("bot")
    result = bot.execute_pipeline(user_question=input_text, index_name="botcon", chat_index_name="botcon-chat")
    
    return result

# Main chat interface
st.title(f"🤖 Bot de Continuonus - Chat with {st.session_state.user_name or 'User'} from {st.session_state.location or 'Location'}")
st.write("""
         Bot de Continuonus is an artificial intelligence that allows you to explore what people participating in the [Carte de Continuous artwork](https://continuon.us/about) entered when asked "What do you want the future to remember?"'
         """)

chat_container = st.container()

with chat_container.chat_message("ai"):
    st.write("What would you like to ask me about what people wrote in the Carte De Continuonus project?")

# Display chat messages and references
with chat_container:
    for entry in st.session_state.chat_data:
        print("this is the entry: ", entry)
        entry_type = entry.get("type")
        content = entry.get("content")
        retrieved_docs = entry.get("retrieved_docs", [])
        past_memory = entry.get("past_memory", [])
        print(f"this is the content: {content}")
        if entry_type == "user":
            with st.chat_message("user"):
                st.write(content)
        elif entry_type == "ai":
            with st.chat_message("ai"):
                st.write(content)
                
                # Display references in expanders
                with st.expander("Memories", expanded=False):
                    with st.expander("Stored Memories", expanded=False):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            with st.expander(f"_\"{doc.page_content}\"_"):
                                st.markdown(f"**Emotions:** {doc.metadata.get('emotions', 'No emotions provided')}")
                                st.markdown(f"**Location:** {doc.metadata.get('location', 'Unknown location')}")
                                st.markdown(f"**Date:** {doc.metadata.get('timestamp_ms', 'Unknown date')}")
                                st.markdown(f"**Sender name:** {doc.metadata.get('sender_name', 'Unknown sender')}")
                    with st.expander("Previous chat", expanded=False):
                        with st.expander(f"_\"{past_memory.page_content}\"_", expanded=False):
                            st.markdown(f"**Date:** {past_memory.metadata.get('date', 'Unknown date')}")
                            st.markdown(f"**User name:** {past_memory.metadata.get('user_name', 'Unknown user name')}")
                            st.markdown(f"**Location:** {past_memory.metadata.get('location', 'Unknown location')}")

# Handle user input
input_text = st.chat_input("Type your message here...")

if input_text:
    try:
        # Append user message to chat_data
        st.session_state.chat_data.append({
            "type": "user",
            "content": input_text,
            "retrieved_docs": []
        })
        
        st.session_state.history.add_user_message(input_text)
        with chat_container.chat_message("user"):
            st.write(input_text)

        with chat_container.chat_message("ai"):
            with st.spinner("Thinking..."):
                # Generate response using the bot
                response = generate_response(input_text)
                answer = response.get("final_answer", "No answer generated")
                initial_bot_response = response.get("initial_bot_response", "No initial bot thought generated")
                retrieved_docs = response.get("initial_context", [])
                past_memory = response.get("past_memory", [])
                
                # Append the AI's response and retrieved documents to the chat_data
                st.session_state.chat_data.append({
                    "type": "ai",
                    "content": answer,
                    "initial_bot_thought": initial_bot_response,
                    "retrieved_docs": retrieved_docs,
                    "past_memory": past_memory
                })

                st.write(answer)

                # Expander for Memories
                with st.expander("Memories", expanded=False):
                    with st.expander("Stored memories", expanded=False):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            with st.expander(f"_\"{doc.page_content}\"_", expanded=False):
                                st.markdown(f"**Location:** {doc.metadata.get('location', 'Unknown location')}")
                                st.markdown(f"**Emotions:** {', '.join(doc.metadata.get('emotions', []))}")
                                st.markdown(f"**Sender:** {doc.metadata.get('sender_name', 'Unknown sender')}")
                # Expander for Previous Chat (Past Memory)
                    with st.expander("Previous chat", expanded=False):
                        with st.expander(f"_\"{past_memory.page_content}\"_", expanded=False):
                            st.markdown(f"**Date:** {past_memory.metadata.get('date', 'Unknown date')}")
                            st.markdown(f"**User name:** {past_memory.metadata.get('user_name', 'Unknown user name')}")
                            st.markdown(f"**Location:** {past_memory.metadata.get('location', 'Unknown location')}")
                            
    except Exception as e:
        st.error(f"Error generating response: {e}")
        print(f"Error generating response: {e}")