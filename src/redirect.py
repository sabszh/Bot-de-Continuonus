import streamlit as st

# Set the target URL
target_url = "https://Bot-de-Continuonus.ploomberapp.io"

st.title("Redirect to the new link for Bot de Continuonus")

# Create a button for redirection
if st.button("Go to EER Chat"):
    st.write(f"Redirecting to [EER Chat]({target_url})...")
    st.markdown(f'<meta http-equiv="refresh" content="0; url={target_url}">', unsafe_allow_html=True)
