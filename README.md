# AskMyPDFs - Your PDF Question Answering Application

## Overview

**AskMyPDFs** is a Streamlit application that empowers you to interact with your PDF documents in a conversational manner.  Powered by the robust capabilities of Google's Gemini models and the Langchain framework, this application allows you to upload PDF files and ask questions about their content, receiving detailed and contextually relevant answers directly from your documents.

This project leverages:

*   **Streamlit:** For creating an interactive and user-friendly web interface.
*   **PyPDF2:** For reading and extracting text from PDF documents.
*   **Langchain:** To orchestrate the language model interaction, text splitting, vector database creation, and question answering chain.
*   **Google Gemini Models (via `langchain-google-genai` and `google-generativeai`):** For generating embeddings to create a vector database and for powering the question answering model.
*   **FAISS (via Langchain):**  For efficient storage and retrieval of document embeddings, enabling fast similarity searches for relevant context.
*   **dotenv:** For managing API keys and environment variables securely.

## Features

*   **Upload Multiple PDFs:** Easily upload and process multiple PDF documents at once.
*   **Intelligent Question Answering:** Ask questions about the content of your uploaded PDFs using a chat-like interface.
*   **Context-Aware Answers:** The application provides answers based on the context extracted from your PDF documents.
*   **Gemini Powered:** Utilizes Google's advanced Gemini models for both embedding generation and question answering, ensuring high-quality and relevant responses.
*   **Local Vector Database:** Creates and saves a local vector database (FAISS index) for efficient retrieval of information from your PDFs.
*   **User-Friendly Interface:**  Built with Streamlit for a simple and intuitive user experience.
*   **Error Handling:**  Includes basic error handling for PDF processing and query execution.

## Installation

To get started with AskMyPDFs, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_DIRECTORY]
    ```
    *(Replace `[YOUR_REPOSITORY_URL]` and `[YOUR_REPOSITORY_DIRECTORY]` with the actual repository URL and directory name)*

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` file yet, create one with the following content based on the libraries used in the code:*

    ```txt
    streamlit
    PyPDF2
    langchain
    langchain-google-genai
    google-generativeai
    faiss-cpu  # or faiss-gpu if you have GPU support
    python-dotenv
    ```
    *Then run `pip install -r requirements.txt`*

4.  **Set up Environment Variables:**
    *   You need a Google Gemini API key. You can obtain one by visiting [Google AI Studio](https://makersuite.google.com/app/apikey).
    *   Create a `.env` file in the root directory of the project.
    *   Add your Gemini API key to the `.env` file like this:
        ```
        GEMINI_API_KEY=YOUR_GEMINI_API_KEY
        ```
        *(Replace `YOUR_GEMINI_API_KEY` with your actual API key)*

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run your_script_name.py  # Replace 'your_script_name.py' with the name of your Python script (likely the name of the provided code, e.g., app.py)
    ```

2.  **Access the application:** Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3.  **Upload PDFs:**
    *   In the sidebar on the left, you will see the "Upload here:" section.
    *   Click on "Browse files" under "Upload Your PDFs Here:" and select the PDF documents you want to process. You can upload multiple PDFs.

4.  **Submit and Process PDFs:**
    *   After uploading your PDFs, click the "Submit and process" button in the sidebar.
    *   Wait for the processing to complete. A "Processing complete!" success message will appear when done. This step creates the vector database from your PDFs. This step needs to be done only once per set of PDFs or when you upload new PDFs.

5.  **Ask Questions:**
    *   In the main chat interface, type your question in the "Ask a question from your pdfs..." input box and press Enter or click the send icon.
    *   The application will process your question, search the vector database for relevant information from your PDFs, and generate an answer using the Gemini model.
    *   The bot's response will be displayed in the chat interface.

## Potential Improvements

*   **Support for more file types:** Extend support to other document formats like DOCX, TXT, etc.
*   **Advanced Prompt Engineering:** Experiment with different prompt templates to refine the question answering and potentially improve answer quality or format.
*   **User Authentication and Sessions:** Implement user authentication to manage different users and their uploaded PDF libraries. Session management would allow users to maintain conversation history.
*   **Deployment:**  Provide instructions and configurations for deploying the application to cloud platforms for wider accessibility.
*   **Enhanced Error Handling and User Feedback:**  Improve error messages and add more informative feedback during PDF processing and question answering.
*   **Summarization Features:** Add functionality to summarize PDF documents or specific sections.
*   **More Control Over Chunking:** Allow users to customize chunk size and overlap for text splitting.


## Contact

Mitesh Nandan

[LinkedIn Profile](https://www.linkedin.com/in/mitesh-nandan/)


---

**Disclaimer:** This application uses Google's Gemini models, which are under active development. The quality and accuracy of responses may vary. Always review the output and exercise your own judgment.
