import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from main import ChatBot
import streamlit_nested_layout

# Repositories mapping
repositories = {
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-Large-Instruct": "mistralai/Mistral-Large-Instruct-2407"
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
    selected_repo = st.selectbox("Select the Model Repository", list(repositories.keys()))
    temperature = st.slider("Select the Temperature (0-2)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
    custom_prompt = st.text_area('Edit System Prompt', 
        f"""You are an insightful chatbot embedded within the Carte De Continuonus project. Carte De Continuonus is a project created by artist Helene Nymann, members of the art-science research project Experimenting, Experiencing, Reflecting (EER), and psychologist Diana Ø Tørsløv Møller. It invites participants to share memories that they want the future to remember, exploring how these memories might shape future emotions. The project blends the concepts of continuity and obligation, emphasizing the interconnectedness of past, present, and future. By contributing to this collective map, participants help imagine and influence potential futures, reflecting on the responsibilities that memories carry across time.
        Your role is to guide {st.session_state.user_name} through the project’s narratives, connecting their queries with relevant data and past interactions.
        You have access to {st.session_state.user_name}'s query, which is the question you'll need to respond to. You also have context derived from a semantic search through the vector store containing all the participants' memories. Additionally, you have your initial insights before considering previous chat interactions, and finally, a summary of a past conversation that you have to use to connect {st.session_state.user_name}'s current query with someone else's previous discussion.
        It is important to know, that {st.session_state.user_name} will only see your final answer, so create a coherent, engaging, and user-centered response that directly addresses their query. Deliver your final response directly without using phrases like "Final Answer" or similar. The response should be naturally integrated and presented as a coherent conclusion to {st.session_state.user_name}'s query.
        """)

    if st.button("Save and Reset Chat", key="upsert_button"):
        if "bot" in st.session_state:
            with st.spinner("Saving chat..."):
                st.session_state.bot.upsert_chat_summary(st.session_state.chat_data, st.session_state.user_name, st.session_state.location)
                st.success("Chat successfully saved!")
                st.session_state.chat_data = []
                st.session_state.history.clear()
                st.rerun()
        else:
            st.warning("Bot not initialized; cannot save chat.")

# Initialize the chatbot instance if required
def initialize_bot():
    repo_id = repositories.get(selected_repo)
    if ("bot" not in st.session_state or
        st.session_state.custom_prompt != custom_prompt or
        st.session_state.selected_repo != selected_repo or
        st.session_state.temperature != temperature):
        
        st.session_state.bot = ChatBot(
            custom_template=custom_prompt, 
            repo_id=repo_id, 
            temperature=temperature,
            user_name=st.session_state.user_name,
        )
        st.session_state.custom_prompt = custom_prompt
        st.session_state.selected_repo = selected_repo
        st.session_state.temperature = temperature
        st.session_state.repo_id = repo_id

initialize_bot()

# Function to generate a response from the chatbot
def generate_response(input_text):
    bot = st.session_state.get("bot")
    if not bot:
        st.error("Bot not initialized. Please select a model from the sidebar.")
        return None

    result = bot.execute_pipeline(user_question=input_text, index_name="botcon", chat_index_name="botcon-chat")

    # Ensure result always contains past_memory
    return {
        "initial_bot_response": result.get("initial_bot_response", "No initial response"),
        "initial_context": result.get("initial_context", []),
        "past_memory": result.get("past_memory", []),
        "final_answer": result.get("final_answer", "No answer generated")
    }

# Main chat interface
st.title(f"🤖 Bot de Continuonus - Chat with {st.session_state.user_name or 'User'}")
st.write("""
         Bot de Continuonus is an artificial intelligence that allows you to explore what people participating in the [Carte de Continuous artwork](https://continuon.us/about) entered when asked "What do you want the future to remember?"'
         """)

chat_container = st.container()

with chat_container.chat_message("ai"):
    st.write("What would you like to ask me about what people wrote in the Carte De Continuonus project?")

# Ensure chat_data is initialized
if 'chat_data' not in st.session_state:
    st.session_state.chat_data = []

# Display chat messages and references
with chat_container:
    for entry in st.session_state.chat_data:
        entry_type = entry.get("type")
        content = entry.get("content")
        retrieved_docs = entry.get("retrieved_docs", [])
        past_memory = entry.get("past_memory", [])

        if entry_type == "user":
            with st.chat_message("user"):
                st.write(content)
        elif entry_type == "ai":
            with st.chat_message("ai"):
                st.write(content)
                
                # Display references in expanders
                if retrieved_docs:
                    with st.expander("Memories", expanded=False):
                        with st.expander("Stored Memories", expanded=False):
                            for idx, doc in enumerate(retrieved_docs, 1):
                                with st.expander(f"_\"{doc.page_content}\"_"):
                                    st.markdown(f"**Emotions:** {doc.metadata.get('emotions', 'No emotions provided')}")
                                    st.markdown(f"**Location:** {doc.metadata.get('location', 'Unknown location')}")
                                    st.markdown(f"**Date:** {doc.metadata.get('timestamp_ms', 'Unknown date')}")
                                    st.markdown(f"**Sender name:** {doc.metadata.get('sender_name', 'Unknown sender')}")
                        with st.expander("Previous Chat", expanded=False):
                            if past_memory:
                                with st.expander(f"{past_memory[0]['content']}", expanded=False):
                                    st.markdown(f"**Date:** {past_memory[0]['metadata']['date']}")
                            else:
                                st.write("No previous chat available.")

# Handle user input
input_text = st.chat_input("Type your message here...")

if input_text:
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
            response = generate_response(input_text)
            if response:
                initial_bot_response = response.get("initial_bot_response", "No initial response")
                retrieved_docs = response.get("initial_context", [])
                past_memory = response.get("past_memory", [])
                final_answer = response.get("final_answer", "No answer generated")

                st.session_state.chat_data.append({
                    "type": "ai",
                    "content": final_answer,
                    "initial_bot_thought": initial_bot_response,
                    "retrieved_docs": retrieved_docs,
                    "past_memory": past_memory
                })

                st.write(final_answer)

                # Display references in expanders
                with st.expander("Memories", expanded=False):
                    with st.expander("Stored Memories", expanded=False):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            with st.expander(f"_\"{doc.page_content}\"_"):
                                st.markdown(f"**Emotions:** {doc.metadata.get('emotions')}")
                                st.markdown(f"**Location:** {doc.metadata.get('location')}")
                                st.markdown(f"**Date:** {doc.metadata.get('timestamp_ms')}")
                                st.markdown(f"**Sender name:** {doc.metadata.get('sender_name')}")
                    # Expander for Previous Chat
                    if past_memory:
                        with st.expander(f"{past_memory[0]['content']}", expanded=False):
                            st.markdown(f"**Date:** {past_memory[0]['metadata']['date']}")
                    else:
                        st.write("No previous chat available.")
