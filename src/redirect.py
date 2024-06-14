import streamlit as st

# Set the target URL
target_url = "https://Bot-de-Continuonus.ploomberapp.io"

# Redirect to the target URL
st.write(f"Redirecting to the new [Bot de Continuonus]({target_url}) link on Ploomber...")
st.markdown(f'<meta http-equiv="refresh" content="0; url={target_url}">', unsafe_allow_html=True)
