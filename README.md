# PWC-RAG
PWC RAG is a easier method to scan through tons of ML research by just hitting a button. A RAG system that mines large amount of information easily. You’ll ask it your questions in natural language and it’ll answer according to relevant papers it finds on [Papers With Code](https://paperswithcode.com/).

On the backend side, the system is powered with a Retrieval Augmented Generation (RAG) framework that relies on a scalable serverless vector database called Upstash, for embeddings we are using BGE models on the HuggingFace, and Mixtral-8x7B-Instruct-v0.1 as LLM from HuggingFace.

On the front-end side, this assistant will be integrated into an interactive and easily deployable web application built with Streamlit.

## Requirements
- HuggingFace API TOKEN
- [Upstash Account](https://upstash.com/docs/common/account/createaccount)
- Upstash URL and TOKEN after creating the index

## Steps to Replicate

1. Clone the repo
    ```bash
      git clone https://github.com/wittyicon29/PWC-RAG.git
    ```
2. Move to the workspace directory
   ```bash
     cd PWC-RAG
   ```
3. Indexing
   To index data into the vector DB, you first need to create an index on Upstash and fill in the credentials in the `.env` file:
   ```
    UPSTASH_URL=...
    UPSTASH_TOKEN=...
   ```
   Run the following command:
   ```bash
     python -m src.index_papers --query "Mistral" --limit 200
   ```
   Result of indexing 200 chunks matching the "Mistral" query.

   ![image](https://github.com/wittyicon29/PWC-RAG/assets/99320225/7d0d3cf6-2ec0-495b-a408-10842852d44c)

   ![upstash-db](https://github.com/wittyicon29/PWC-RAG/assets/99320225/02eadacc-08d1-46ea-8675-bb07b909e7c4)

4. Running the streamlit applications locally

   Before running the streamlit app, you have to set the Huggingface API token in the '.env' file:
   ```bash
     HUGGINGFACE_API_TOKEN=...
   ```
   Now you can the streamlit app
   ```bash
     python -m streamlit run  src/app.py
   ```
   
   ![cast](https://github.com/wittyicon29/PWC-RAG/assets/99320225/879b8db8-8e21-49be-aa1f-4708d928a366)

## Notes 
 - You can use any Embedding model supported by [Langchain](https://python.langchain.com/docs/integrations/text_embedding)
 - You can try different LLMs on [Langchain](https://python.langchain.com/docs/integrations/llms/) to evaluate the RAG system
 - The app is ready to deploy on Google Cloud Run using docker or Streamlit cloud.
