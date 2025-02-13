from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai

import numpy as np
import streamlit as st
import faiss
import os
import logging
import time
import socket

if "messages" not in st.session_state:
    st.session_state["messages"] = []


logging.basicConfig(filename = "app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

def is_localhost():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        local_ips = ['127.0.0.1', 'localhost']

        return host_ip.startswith("192.168.") or host_ip.startswith("10.") or host_ip in local_ips
    except Exception as e:
        return False

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

class PdfChatbot():
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY') if is_localhost() else st.secrets['GEMINI_API_KEY']
        # self.api_key = st.secrets['GEMINI_API_KEY']
        self.client = genai.Client(api_key=self.api_key)
        # self.vector_database = None
        self.documents = None
        self.embeddings = None

    def read_pdf(self, pdfs):
        if len(pdfs) == 0:
            # st.error("Please Upload PDF")
            raise Exception("No pdf uploaded")
        else:        
            text = ""
            for pdf in pdfs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
    
    def split_text(self,text:str):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
        self.documents = text_splitter.split_text(text)
        st.session_state["documents"] = self.documents
        return text_splitter.split_text(text)
    
    def generate_embeddings(self,texts:list):
        embeddings = []
        for text in texts:
            request = self.client.models.embed_content(
                model = "text-embedding-004",
                contents= text
            )
            if request.embeddings is not None:
                embeddings.append(request.embeddings[0].values)
        self.embeddings = embeddings
        logging.info("created text embeddings successfully")
        return embeddings
    
    def generate_vector_db(self, embeddings):
        try:
            # Ensure embeddings are numpy array and of type float32
            embeddings = np.array(embeddings, dtype=np.float32)
            
            if len(embeddings.shape) != 2:
                raise ValueError(f"Expected 2D embeddings array, but got shape: {embeddings.shape}")

            dim = embeddings.shape[1]  # Get embedding dimension
            index = faiss.IndexFlatL2(dim)  # Initialize FAISS index

            index.add(embeddings) #type:ignore # Add embeddings to FAISS index
            faiss.write_index(index, "db.index")
            # self.vector_database = index
            logging.info("Created vector database successfully")
            return True

        except Exception as e:
            logging.error(f"Error creating vector database: {e}")
            return None  # Return None to avoid breaking the app

    
    def user_input(self, question:str):
        st.session_state["messages"].append({"role": "user", "content": question})
        request = self.client.models.embed_content(
            model="text-embedding-004",
            contents= question
        )
        index = faiss.read_index('db.index')
        if request.embeddings is not None:
            query_embedding = request.embeddings[0].values
            logging.info("embeded user query")
            # print(query_embedding)
            if index is not None:
                distances, indices = index.search(np.array([query_embedding], dtype=np.float32), 4)
                self.documents = st.session_state.get("documents", None)
                docs = []
                if indices is not None and len(indices) > 0:
                    for idx in indices[0]:  # Ensure we access the first row of results
                        if idx != -1 and self.documents is not None:  # Skip invalid indices
                            docs.append(self.documents[idx]) 
                context = " ".join(docs)
                # print(docs)
                # print(self.documents)
                prompt = f'''Answer the questions in as detail as possible from the provided context. Make sure
    to provide all the details from the context, if the answer is not in the context,
    just say 'answer is not in the provided pdfs', dont provide any wrong answers.
    
    context: {context}\n
    question: {question}
    '''
                response = self.client.models.generate_content(
                    model= 'gemini-2.0-flash',
                    contents= [prompt],
                )
                st.session_state["messages"].append({"role": "bot", "content": response.text})
                for i, message in enumerate(st.session_state["messages"]):
                    with st.chat_message(message["role"]):
                        if message["role"] == "user":
                            st.markdown(message["content"])
                        else:
                            if i == len(st.session_state["messages"]) - 1:  # Only stream the last message
                                stream = stream_data(message["content"])
                                st.write_stream(stream)
                            else:
                                st.markdown(message["content"])
                # with st.chat_message("bot"):
                #     st.markdown(response.text)
            else:
                raise ValueError("Vector database is not initialized.")
        pass
        # distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
        # return indices[0]

def main():
    logging.info("starting App...")
    st.set_page_config("AskMyPDFs", layout="wide")
    st.title("AskMyPDFs")

    user_question = st.chat_input("Ask a question from your pdfs...")

    if user_question:
        # pass
        chatbot.user_input(user_question)

    with st.sidebar:
        st.title("Upload here:")
        pdf_docs = st.file_uploader("Upload Your PDFs Here:", accept_multiple_files=True)
        if st.button("Submit and process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = chatbot.read_pdf(pdf_docs)
                    logging.info("processed PDFs to text successfully")

                    text_chunks = chatbot.split_text(raw_text)
                    logging.info("Created text chunks successfully")

                    chatbot.generate_embeddings(text_chunks)
                    logging.info("created embeddings")

                    chatbot.generate_vector_db(chatbot.embeddings)
                    logging.info("created vector db successfully")

                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

        st.markdown("---")
        st.markdown("## 🧑🏾‍🦱 Mitesh Nandan")
        st.markdown("Connect with me on LinkedIn:")
        linkedin_url = "https://www.linkedin.com/in/mitesh-nandan/"
        git_profile =  "https://github.com/mforker"
        st.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)]({linkedin_url})")
        st.markdown(f"[![GitHub Profile](https://img.shields.io/badge/GitHub-mforker-blue?logo=github)]({git_profile})")

chatbot = PdfChatbot()
if __name__ == '__main__':
    main()

        