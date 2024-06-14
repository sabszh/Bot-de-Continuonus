from main import ChatBot
import streamlit as st

# Repositories 
repositories = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1"
}

st.set_page_config(page_title="Bot de Continuonus")

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

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
if "first_question" not in st.session_state:
    st.session_state.first_question = ""

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
    result = bot.rag_chain.invoke(input)
    return result

# Main container for chat messages
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Handle user input at the bottom
input = st.chat_input("Type your message here...")

if input:
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Store the first question if not already set
    if not st.session_state.first_question:
        st.session_state.first_question = input

    # Generate response if needed
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating an answer..."):
                response = generate_response(input)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
