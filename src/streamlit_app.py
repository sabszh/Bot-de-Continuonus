from main import ChatBot
import streamlit as st

repositories = {
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1"
}

retriever_methods = {
    "MultiQueryRetriever": "multiquery_retriever_llm",
    "Vector store-backed retriever": "docsearch"
}

st.set_page_config(page_title="Bot de Continuonus")
with st.sidebar:
    st.title('Bot de Continuonus')
    st.write("Be aware, if you make any changes here that the chatbot will reload and your chat will be gone.")
    selected_repo = st.selectbox("Select the Model Repository", list(repositories.keys()))
    selected_retriever = st.selectbox("Select the Retriever Method", list(retriever_methods.keys()))
    temperature = st.slider("Select the Temperature (0-2)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
    custom_prompt = st.text_area('Edit System Prompt',
    """You are a clairvoyant chatbot who bridges depths of collective pasts and future possibilities.
    Rooted in the Carte De Continuonus project, you're here to field questions about how individuals envision their memories shaping the future.
    Drawing from the innovative collaboration of art, science, and psychology, you provide insights into the collective tapestry of emotions and aspirations.
    Ready to guide users through their journey of envisioning and reflecting on the future.""",height=250)

# Initialize ChatBot based on selected repository and temperature
@st.cache(allow_output_mutation=True)
def load_model():
    return ChatBot(repo_id=st.session_state.repo_id, temperature=st.session_state.temperature, retriever_method=st.session_state.retriever_method)

# Initialize ChatBot based on selected repository and temperature
if "bot" not in st.session_state.keys() or st.session_state.custom_prompt != custom_prompt or st.session_state.selected_repo != selected_repo or st.session_state.temperature != temperature or "selected_retriever" not in st.session_state.keys() or st.session_state.selected_retriever != selected_retriever:
    # Determine repo_id based on selected_repo
    repo_id = repositories[selected_repo]
    
    # Determine retriever_method based on selected_retriever
    retriever_method = retriever_methods[selected_retriever]
    
    # Store repo_id and retriever_method in session_state
    st.session_state.repo_id = repo_id
    st.session_state.retriever_method = retriever_method
    
    # Initialize ChatBot with repo_id and retriever_method
    bot = ChatBot(custom_template=custom_prompt, repo_id=repo_id, temperature=temperature, retriever_method=retriever_method)
    
    # Store bot and other parameters in session_state
    st.session_state.bot = bot
    st.session_state.custom_prompt = custom_prompt
    st.session_state.selected_repo = selected_repo
    st.session_state.temperature = temperature
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
    st.session_state.selected_retriever = selected_retriever
else:
    bot = st.session_state.bot

# Function for generating LLM response
def generate_response(messages):
    input_text = "\n".join(message["content"] for message in messages)
    result = bot.rag_chain.invoke(input_text)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating an answer.."):
            response = generate_response(st.session_state.messages)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
