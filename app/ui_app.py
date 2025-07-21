import streamlit as st
from rag_main import answer_question

st.set_page_config(page_title="RAG Question Answering", layout="centered")
st.title("📘 ss-22-bot")

st.markdown("This app uses Retrieval-Augmented Generation (RAG) to answer questions based on your custom data.")

query = st.text_input("🧠 Ask a question:", placeholder="e.g., What are the fundamental rights in India?")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = answer_question(query)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
