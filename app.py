import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api = st.secrets['GEMINI_API_KEY']
# os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing PDF: {pdf.name}, {str(e)}")
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_db(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_db.save_local("faiss_index")
        # st.success("Vector database created successfully!")
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")

def get_conversation_chain():
    prompt_temp = """
    Answer the questions in as detail as possible from the provided context. Make sure
    to provide all the details from the context, if the answer is not in the context,
    just say 'answer is not in the context', dont provide any wrong answers.
    context: \n {context}\n
    question: \n {question}\n

    answer: """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_temp, input_variables=['context', 'question'])
    
    llm_chain = LLMChain(llm=model, prompt=prompt)
    # llm_chain = prompt | model
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    
    
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversation_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        with st.chat_message('bot'):
            st.markdown(response["output_text"])
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def main():
    st.set_page_config("AskMyPDFs", layout="wide")
    st.title("AskMyPDFs")

    user_question = st.chat_input("Ask a question from your pdfs...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload here:")
        pdf_docs = st.file_uploader("Upload Your PDFs Here:", accept_multiple_files=True)
        if st.button("Submit and process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_chunks(raw_text)
                    get_vector_db(text_chunks)
                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

if __name__ == "__main__":
    main()