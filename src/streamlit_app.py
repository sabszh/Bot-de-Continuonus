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
    st.title('Bot de Continuonus')
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
    history.add_ai_message("Ask me about the future!")

# Initialize document history if not present
if "documents" not in st.session_state:
    st.session_state.documents = []

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

# Add custom CSS to fix the input field at the bottom
st.markdown("""
    <style>
    .stTextInput, .stButton {
        position: fixed;
        bottom: 10px;
        width: calc(100% - 20px);
        margin-left: 10px;
        margin-right: 10px;
    }
    .stButton { 
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Create two columns: one for chat, one for sources
chat_col, sources_col = st.columns([2, 1])

# Main container for chat messages
with chat_col:
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in history.messages:
            st.chat_message(message.type).write(message.content)

# Handle user input at the bottom
input = st.chat_input("Type your message here...")

if input:
    history.add_user_message(input)
    with chat_col.chat_message("user"):
        st.write(input)

    # Generate response if needed
    if history.messages[-1].type != "ai":
        with chat_col.chat_message("ai"):
            with st.spinner("Generating an answer..."):
                response = generate_response(input)
                answer = response["answer"]
                retrieved_docs = response["retrieved_docs"]

                st.write(answer)
                history.add_ai_message(answer)
                st.session_state.documents.insert(0, {
                    "answer": answer,
                    "retrieved_docs": retrieved_docs
                })

# Display sources in the right column
with sources_col:
    if st.session_state.documents:
        tabs = st.tabs([f"Message {len(st.session_state.documents) - i}" for i in range(len(st.session_state.documents))])
        for i, tab in enumerate(tabs):
            with tab:
                doc_history = st.session_state.documents[i]
                for idx, doc in enumerate(doc_history["retrieved_docs"], 1):
                    with st.expander(f"Source {idx}"):
                        st.write(f"**Content:** {doc.page_content}")
                        st.write(f"**Emotions:** {doc.metadata['emotions']}")
                        st.write(f"**Location:** {doc.metadata['location']}")
    else:
        st.write("Sources will appear here after you ask a question...")
