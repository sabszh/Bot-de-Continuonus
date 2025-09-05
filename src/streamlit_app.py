import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import uuid
from main import chatbot
import streamlit_nested_layout

st.set_page_config(page_title="Bot de Continuonus", layout="wide")

# Initialize session state variables
if "chat_data" not in st.session_state:
    st.session_state.chat_data = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = StreamlitChatMessageHistory()

if "future_wish" not in st.session_state:
    st.session_state.future_wish = None

# =============================
# Helpers: TTS controls
# =============================
def tts_controls(text: str, key: str = "tts"):
    if not text:
        return
    escaped = (
        text.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", " ")
    )
    js = f"""
    <script>
      var utterance_{key} = new SpeechSynthesisUtterance('{escaped}');
      utterance_{key}.rate = 0.95;
      utterance_{key}.pitch = 1.1;
      var synth_{key} = window.speechSynthesis;

      function play_{key}() {{
        if (!synth_{key}.speaking) {{
          synth_{key}.speak(utterance_{key});
        }} else {{
          synth_{key}.resume();
        }}
      }}
      function pause_{key}() {{
        if (synth_{key}.speaking) synth_{key}.pause();
      }}
    </script>
    <div style="display:flex;gap:8px;align-items:center;margin-top:6px;">
      <button onclick="play_{key}()">▶️ Play</button>
      <button onclick="pause_{key}()">⏸️ Pause</button>
    </div>
    """
    st.components.v1.html(js, height=50)

# =============================
# User info dialogs
# =============================
@st.dialog("User Name and Location", width="small")
def ask_name():
    user_name = st.text_input("Please enter your name:")
    location = st.text_input("Please enter your location:")
    if st.button("Submit"):
        if user_name and location:
            st.session_state.user_name = user_name
            st.session_state.location = location
            st.rerun()

@st.dialog("What should the future remember?", width="small")
def ask_wish():
    wish = st.text_area("Share your wish (you can keep it brief)")
    if st.button("Save wish"):
        if wish:
            st.session_state.future_wish = wish
            st.rerun()

if st.session_state.user_name is None:
    ask_name()
elif st.session_state.get("future_wish") is None:
    ask_wish()

# =============================
# Bot init
# =============================
if "bot" not in st.session_state:
    st.session_state.bot = chatbot(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.8,
        user_name=st.session_state.user_name,
        session_id=st.session_state.session_id,
    )

# =============================
# Response generator
# =============================
def generate_response(input_text: str):
    bot = st.session_state.get("bot")

    chat_pairs = []
    for msg in st.session_state.chat_data:
        if msg.get("type") == "user":
            chat_pairs.append(f"User: {msg.get('input_text','')}")
        elif msg.get("type") == "ai":
            chat_pairs.append(f"AI: {msg.get('ai_output','')}")
    chat_history = "\n".join(chat_pairs)

    effective_input = input_text
    if st.session_state.get("future_wish"):
        effective_input = (
            f"{input_text}\n\nUser's wish for what the future should remember: \"{st.session_state.future_wish}\"."
        )

    result = bot.pipeline(
        user_input=effective_input,
        user_name=st.session_state.user_name,
        user_location=st.session_state.location,
        session_id=st.session_state.session_id,
        chat_history=chat_history,
    )
    return result

# =============================
# UI
# =============================
st.title("🤖 Bot de Continuonus")
st.write("""
Bot de Continuonus is an artificial intelligence that lets you explore what people participating in 
the [Carte de Continuonus artwork](https://continuon.us/about) entered when asked: 
*"What do you want the future to remember?"*
""")

chat_container = st.container()

with chat_container.chat_message("ai"):
    st.write(f"Hi {st.session_state.user_name}, what would you like to ask me about what people wrote in the Carte De Continuonus project?")

with chat_container:
    for entry in st.session_state.chat_data:
        entry_type = entry.get("type")
        if entry_type == "user":
            with st.chat_message("user"):
                st.write(entry["input_text"])
        elif entry_type == "ai":
            with st.chat_message("ai"):
                st.write(entry["ai_output"])
                tts_controls(entry["ai_output"], key=f"tts_{len(st.session_state.chat_data)}")

                source_data = entry.get("source_data", [])
                past_chat_context = entry.get("past_chat_context", [])

                with st.expander("Referenced data", expanded=False):
                    with st.expander("Submissions to the Continuonus Artwork", expanded=False):
                        for idx, doc in enumerate(source_data, 1):
                            text = doc.get("text", "")
                            metadata = doc.get("metadata", {})
                            with st.expander(f"_\"{text}\"_", expanded=False):
                                st.markdown(f"**Sender:** {metadata.get('sender_name', 'Unknown sender')}")
                                st.markdown(f"**Location:** {metadata.get('location', 'Unknown location')}")
                                st.markdown(f"**Date:** {metadata.get('date', 'Unknown date')}")

                    if past_chat_context:
                        with st.expander("Data from previous conversations with this LLM", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                metadata = doc.get("metadata", {})
                                with st.expander(f"User question: _\"{metadata.get('user_question', 'Unknown question')}\"_", expanded=False):
                                    st.markdown(f"**AI Response:** {metadata.get('ai_output', 'Unknown response')}")
                                    st.markdown(f"**Date:** {metadata.get('date', 'Unknown date')}")
                                    st.markdown(f"**User name:** {metadata.get('user_name', 'Unknown user')}")
                                    st.markdown(f"**Location:** {metadata.get('location', 'Unknown location')}")

# --- Handle input ---
input_text = st.chat_input("Type your message here...")

if input_text:
    try:
        st.session_state.chat_data.append({
            "type": "user",
            "user_name": st.session_state.user_name,
            "input_text": input_text,
            "session_id": st.session_state.session_id,
        })

        with chat_container.chat_message("user"):
            st.write(input_text)

        with st.spinner("Thinking..."):
            result = generate_response(input_text)
            ai_output = result.get("ai_output", "No answer generated")
            source_data = result.get("source_data", [])
            past_chat_context = result.get("past_chat_context", [])

            st.session_state.chat_data.append({
                "type": "ai",
                "ai_output": ai_output,
                "source_data": source_data,
                "past_chat_context": past_chat_context
            })

            with chat_container.chat_message("ai"):
                st.write(ai_output)
                tts_controls(ai_output, key=f"tts_{len(st.session_state.chat_data)}")

                with st.expander("Referenced data", expanded=False):
                    with st.expander("Submissions to the Continuonus Artwork", expanded=False):
                        for idx, doc in enumerate(source_data, 1):
                            text = doc.get("text", "")
                            metadata = doc.get("metadata", {})
                            with st.expander(f"_\"{text}\"_", expanded=False):
                                st.markdown(f"**Sender:** {metadata.get('sender_name', 'Unknown sender')}")
                                st.markdown(f"**Location:** {metadata.get('location', 'Unknown location')}")
                                st.markdown(f"**Date:** {metadata.get('date', 'Unknown date')}")

                    if past_chat_context:
                        with st.expander("Data from previous conversations with this LLM", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                metadata = doc.get("metadata", {})
                                with st.expander(f"User question: _\"{metadata.get('user_question', 'Unknown question')}\"_", expanded=False):
                                    st.markdown(f"**AI Response:** {metadata.get('ai_output', 'Unknown response')}")
                                    st.markdown(f"**Date:** {metadata.get('date', 'Unknown date')}")
                                    st.markdown(f"**User name:** {metadata.get('user_name', 'Unknown user')}")
                                    st.markdown(f"**Location:** {metadata.get('location', 'Unknown location')}")
    except Exception as e:
        st.error(f"Error generating response: {e}")
