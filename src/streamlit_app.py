from main import ChatBot
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Repositories 
repositories = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1"
}

st.set_page_config(page_title="Bot de Continuonus", layout="wide")

# Sidebar configuration
with st.sidebar:
    st.title('LLM Settings')
    st.write("""Be aware, if you make any changes here that the chatbot will reload and your chat will be gone.
             If you get an error, try a different model. If that does not work, it might be overloaded or down - so try again later.""")
    selected_repo = st.selectbox("Select the Model Repository", list(repositories.keys()))
    temperature = st.slider("Select the Temperature (0-2)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
    custom_prompt = st.text_area('Edit System Prompt',
    """You are a clairvoyant chatbot who bridges depths of collective pasts and future possibilities.
    Rooted in the Carte De Continuonus project, you're here to field questions about how individuals envision their memories shaping the future.
    Drawing from the innovative collaboration of art, science, and psychology, you provide insights into the collective tapestry of emotions and aspirations.
    Ready to guide users through their journey of envisioning and reflecting on the future.
    Don't include any questions stated from the RAG-chain.
    Only answer the user question, but include the contexts given.""", height=250)

# Initialize message history
history = StreamlitChatMessageHistory(key="chat_messages")

# Add initial message if it's the first time loading
if len(history.messages) == 0:
    history.add_ai_message("What would you like to ask me about what people wrote in the Carte De Continuonus project?")

# Initialize a container to hold both AI and user messages along with their references
if "chat_data" not in st.session_state:
    st.session_state.chat_data = []

# Add initial AI prompt message to the state if it's not already there
if len(st.session_state.chat_data) == 0:
    st.session_state.chat_data.append({
        "type": "ai",
        "content": "What would you like to ask me about what people wrote in the Carte De Continuonus project?",
        "retrieved_docs": []  # No references for the initial prompt
    })

# Initialize ChatBot instance if needed
def initialize_bot():
    repo_id = repositories[selected_repo]
    bot = ChatBot(custom_template=custom_prompt, repo_id=repo_id, temperature=temperature)
    st.session_state.bot = bot
    st.session_state.custom_prompt = custom_prompt
    st.session_state.selected_repo = selected_repo
    st.session_state.temperature = temperature
    st.session_state.repo_id = repo_id

if "bot" not in st.session_state or st.session_state.custom_prompt != custom_prompt or st.session_state.selected_repo != selected_repo or st.session_state.temperature != temperature:
    initialize_bot()
else:
    bot = st.session_state.bot

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain(input, return_docs=True)
    return result

# Main container for chat messages
st.title("Bot de Continuonus")
st.write('Bot de Continuonus is an artificial intelligence that allows you to explore what people participating in the Continuous artwork entered when asked "What do you want the future to remember?"')
chat_container = st.container()

# Display chat messages and references
with chat_container:
    for entry in st.session_state.chat_data:
        if entry["type"] == "user":
            st.chat_message("user").write(entry["content"])
        elif entry["type"] == "ai":
            st.chat_message("ai").write(entry["content"])
            if entry["retrieved_docs"]:
                with st.expander("References", expanded=False):  # Keep the expander closed by default
                    for idx, doc in enumerate(entry["retrieved_docs"], 1):
                        st.markdown(f"**Document {idx}:** {doc.page_content}")
                        st.markdown(f"**Emotions:** {doc.metadata['emotions']}")
                        st.markdown(f"**Location:** {doc.metadata['location']}")
                        st.markdown(f"**Date:** {doc.metadata['timestamp_ms']}")
                        st.markdown(f"**Sender name:** {doc.metadata['sender_name']}")

# Handle user input at the bottom
input = st.chat_input("Type your message here...")

if input:
    # Store user message
    st.session_state.chat_data.append({
        "type": "user",
        "content": input,
        "retrieved_docs": []
    })
    history.add_user_message(input)
    with chat_container.chat_message("user"):
        st.write(input)

    # Generate response if needed
    if history.messages[-1].type != "ai":
        with chat_container.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = generate_response(input)
                answer = response["answer"]
                retrieved_docs = response["retrieved_docs"]

                # Store both the AI response and its references together
                st.session_state.chat_data.append({
                    "type": "ai",
                    "content": answer,
                    "retrieved_docs": retrieved_docs
                })

                # Display the AI response immediately
                st.write(answer)

                # Display the associated references immediately
                if retrieved_docs:
                    with st.expander("References", expanded=False):  # Keep the expander closed by default
                        for idx, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"**Document {idx}:** {doc.page_content}")
                            st.markdown(f"**Emotions:** {doc.metadata['emotions']}")
                            st.markdown(f"**Location:** {doc.metadata['location']}")
                            st.markdown(f"**Date:** {doc.metadata['timestamp_ms']}")
                            st.markdown(f"**Sender name:** {doc.metadata['sender_name']}")
                else:
                    st.info("No references available for this response.")
