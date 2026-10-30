import streamlit as st

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon=None,
)

st.title("Medical RAG Assistant")
st.write("Upload a medical document and ask questions about its contents.")

uploaded_file = st.file_uploader(
    "Upload a medical PDF",
    type=["pdf"],
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    question = st.text_input("Ask a question about the document")

    if question:
        st.info("The RAG pipeline will be connected here.")