import os
import time
from typing import List
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.rag import RAG

st.set_page_config(layout="wide")
load_dotenv()


left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('assets\logo.png', width=600)
    


st.header('Papers with Code RAG', divider='rainbow')

@st.cache_resource
def get_embedding_model():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings


def load_rag(_chat_box):
    embeddings = get_embedding_model()
    rag = RAG(_chat_box, embeddings)
    return rag


def display_source_documents(source_documents):
    for document, score in source_documents:
        metadata = document.metadata
        document_content = document.page_content

        id_ = metadata["id"]
        arxiv_id = metadata["arxiv_id"]
        url_pdf = metadata["url_pdf"]
        title = metadata["title"]
        authors = metadata["authors"]
        published = metadata["published"]

        with st.container(border=True):
            st.markdown(f"* **📰 Title** : {title} (score = {score})")
            st.markdown(f"* **🏷️ ARXIV ID** : **`{arxiv_id}`**")
            st.markdown(f"* **✍️ Authors** : {' ,'.join(authors)}")
            st.markdown(f"* **📅 Publication date** : {published}")
            st.markdown(f"URL 🔗: {url_pdf}")
            st.write(f"context: {document_content}")



input_question = st.text_input("Ask your question")
columns = st.columns(2)

with columns[0]:
    chat_box = st.empty()

rag = load_rag(chat_box)


if st.button('Hit me'):
    if input_question.strip() != "":
        with st.spinner("Generating Answer"):
            prediction = rag.predict(input_question)

        answer = prediction["answer"]
        source_documents = prediction["source_documents"]       
        with columns[1]:
            st.write("### Source documents")
            display_source_documents(source_documents)
else:
    st.write("Please enter a question first.")
