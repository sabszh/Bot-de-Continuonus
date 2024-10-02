import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import uuid
from main import chatbot
import streamlit_nested_layout

# Repositories mapping
repositories = {
    "Llama 70B Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Mistral 8x7b Instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.1"
}

st.set_page_config(page_title="Bot de Continuonus", layout="wide", initial_sidebar_state='collapsed')

# Initialize session state variables
if "chat_data" not in st.session_state:
    st.session_state.chat_data = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "session_id" not in st.session_state:
    # Generate a random session ID using uuid
    st.session_state.session_id = str(uuid.uuid4())

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
        """Be aware, if you make any changes here, the chatbot will reload, and your chat will be lost.
        If you encounter errors, try a different model or check if the model is overloaded. Try again later if needed."""
    )

    # Model and temperature settings
    selected_repo = st.selectbox("Select the Model Repository", list(repositories.keys()))
    temperature = st.slider("Select the Temperature (0-2)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)

    # Sourcedata prompt editing
    prompt_sourcedata = st.text_area(
        'Edit Prompt for Sourcedata, use these variables: {chat_history} {user_input}, {orginal_data}, {user_name}',
       """You are an assistant associated with the artwork Carte de Continuonus by Helena Nymann. This project invites people viewing the artwork to respond to the question: "What do you want the future to remember?" The data consists of their name, location, and the response they entered to the question, which is the data we are most interested in. You are now assisting the user "{user_name}" with their query: "{user_input}". Below is the relevant data submitted to the continuonus artwork that was retrieved from the database for this query: "{original_data}". Based on this data, provide a concise answer to the user’s question in 1-3 sentences. The previous chat history for this session so far is: {chat_history} Your response:"""
)

    # Conversation prompt editing
    prompt_conv = st.text_area(
        'Edit Prompt for Conversation Context, use these variables:{chat_history} {user_input}, {llm_response}, {past_chat}, {user_name}',
        """You are an assistant observing conversations between the user and another LLM regarding statements submitted by viewers of the “Carte de Continuonus” artwork, by Helene Nyman. All interactions between people and the LLM are recorded and stored in your database. When people ask questions about the data, you get the question and the answer from the LLM. You use that data to search your database of past conversations for conversations that might be related. You will create a summary of those past conversations no longer than 4 sentences. Your summary should mention the name of the person involved in the past conversations.
        Here is the last question asked by the user in this session: "{user_name}" asked: {user_input}
        Here is what the LLM you are watching responded with and what the user is seeing: "{llm_response}"
        Here is relevant data from past conversations that is relevant: {past_chat}
        Here is the chat history for this session, so that your response can be aware of the context: {chat_history}
        Your response:""")

# Initialize the chatbot instance if required
def initialize_bot():
    repo_id = repositories.get(selected_repo)
    
    try:
        if ("bot" not in st.session_state or
            st.session_state.prompt_sourcedata != prompt_sourcedata or
            st.session_state.prompt_conv != prompt_conv or
            st.session_state.selected_repo != selected_repo or
            st.session_state.temperature != temperature):
            st.session_state.bot = chatbot( 
                repo_id=repo_id, 
                temperature=temperature,
                prompt_sourcedata=prompt_sourcedata,
                prompt_conv=prompt_conv,
                user_name=st.session_state.user_name,
                session_id=st.session_state.session_id,
            )
            st.session_state.prompt_sourcedata = prompt_sourcedata
            st.session_state.prompt_conv = prompt_conv
            st.session_state.selected_repo = selected_repo
            st.session_state.temperature = temperature
            st.session_state.repo_id = repo_id
            st.session_state.session_id = st.session_state.session_id
            
    except Exception as e:
        st.error(f"Error initializing bot: {e}")
        print(f"Error initializing bot: {e}")

initialize_bot()

# Function to generate a response from the chatbot
def generate_response(input_text):
    bot = st.session_state.get("bot")
    
    # Include chat history in the response generation
    chat_history = "\n".join([
        f"User: {msg.get('input_text', '')}\nAI: {msg.get('ai_output', '')}" 
        for msg in st.session_state.chat_data 
        if msg.get('type') == 'ai' or msg.get('type') == 'user'
        ])
    
    result = bot.pipeline(user_input=input_text, user_name=st.session_state.user_name, user_location=st.session_state.location, session_id=st.session_state.session_id, chat_history=chat_history)
    
    return result

# Main chat interface
st.title(f"🤖 Bot de Continuonus")
st.write("""
         Bot de Continuonus is an artificial intelligence that allows you to explore what people participating in the [Carte de Continuonus artwork](https://continuon.us/about) entered when asked "What do you want the future to remember?"'
         """)

chat_container = st.container()

with chat_container.chat_message("ai"):
    st.write(f"Hi {st.session_state.user_name}, what would you like to ask me about what people wrote in the Carte De Continuonus project?")

# Display all previous chat messages
with chat_container:
    for entry in st.session_state.chat_data:
        entry_type = entry.get("type")
        ai_output = entry.get("ai_output")
        user_input = entry.get("input_text", "")
        source_data = entry.get("source_data", [])
        past_chat_context = entry.get("past_chat_context", [])

        if entry_type == "user":
            with st.chat_message("user"):
                st.write(user_input)  # Display user message
        elif entry_type == "ai":
            with st.chat_message("ai"):
                st.write(ai_output)  # Display AI response

                # Expander for Memories (Referenced Data)
                with st.expander("Referenced data", expanded=False):
                    with st.expander("Submissions to the Continuonus Artwork", expanded=False):
                        for idx, doc in enumerate(source_data, 1):
                            with st.expander(f"_\"{doc.page_content}\"_", expanded=False):
                                st.markdown(f"**Sender:** {doc.metadata.get('sender_name', 'Unknown sender')}") 
                                st.markdown(f"**Location:** {doc.metadata.get('location', 'Unknown location')}")
                                st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")

                    # Expander for Previous Chat (Past Memory)
                    if past_chat_context:
                        with st.expander("Data from previous conversations with this LLM", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                with st.expander(f"User question: _\"{doc.metadata.get('user_question')}\"_", expanded=False):
                                    st.markdown(f"**AI Response:** {doc.metadata.get('ai_output')}")
                                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")
                                    st.markdown(f"**User name:** {doc.metadata.get('user_name', 'Unknown user name')}")
                                    st.markdown(f"**Location:** {doc.metadata.get('location', 'Unknown location')}")

# Handle user input
input_text = st.chat_input("Type your message here...")

if input_text:
    try:
        # Append user message to chat_data
        st.session_state.chat_data.append({
            "type": "user",
            "user_name": st.session_state.user_name,
            "input_text": input_text,
            "session_id": st.session_state.session_id,
            "retrieved_docs": []
        })
        
        # Display user message immediately
        with chat_container.chat_message("user"):
            st.write(input_text)

        # Generate AI response
        with st.spinner("Thinking..."):
            result = generate_response(input_text)
            ai_output = result.get("ai_output", "No answer generated")
            source_data = result.get("source_data", [])
            past_chat_context = result.get("past_chat_context", [])

            # Append AI response to chat_data
            st.session_state.chat_data.append({
                "type": "ai",
                "ai_output": ai_output,
                "source_data": source_data,
                "past_chat_context": past_chat_context
            })

            # Display AI message
            with chat_container.chat_message("ai"):
                st.write(ai_output)
                
                # Display referenced data
                with st.expander("Referenced data", expanded=False):
                    with st.expander("Submissions to the Continuonus Artwork", expanded=False):
                        for idx, doc in enumerate(source_data, 1):
                            with st.expander(f"_\"{doc.page_content}\"_", expanded=False):
                                st.markdown(f"**Sender:** {doc.metadata.get('sender_name', 'Unknown sender')}") 
                                st.markdown(f"**Location:** {doc.metadata.get('location', 'Unknown location')}")
                                st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")

                    # Display past chat context if available
                    if past_chat_context:
                        with st.expander("Data from previous conversations with this LLM", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                with st.expander(f"User question: {doc.metadata.get('user_question')}", expanded=False):
                                    st.markdown(f"**AI Response:** {doc.metadata.get('ai_output')}")
                                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")
                                    st.markdown(f"**User name:** {doc.metadata.get('user_name', 'Unknown user name')}")
                                    st.markdown(f"**Location:** {doc.metadata.get('location', 'Unknown location')}")
                                    
    except Exception as e:
            st.error(f"Error generating response: {e}")
