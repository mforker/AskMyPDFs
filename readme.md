# AskMyPDFs

## 📌 Overview
AskMyPDFs is an AI-powered chatbot that allows users to upload PDFs and ask questions about their content. The app processes the PDFs, generates embeddings, stores them in a vector database, and retrieves relevant information based on user queries.

## 🚀 Features
- 📂 Upload and process PDFs
- 🔍 Semantic search using vector embeddings
- 🤖 AI-powered chatbot with context-based responses
- 🗂️ FAISS-based vector storage for fast retrieval
- 🌐 Streamlit-based interactive UI

## 🏗️ Tech Stack
- **Backend**: Python, FAISS, OpenAI Gemini API (or Google GenAI API)
- **Frontend**: Streamlit
- **Database**: FAISS for vector storage
- **Libraries**: `numpy`, `faiss`, `streamlit`, `logging`, `requests`

## 📥 Installation
```sh
# Clone the repository
git clone https://github.com/mforker/AskMyPDFs.git
cd AskMyPDFs

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Usage
```sh
streamlit run app.py
```

## ⚡ How It Works
1. User uploads a PDF.
2. The app extracts text from the PDF and creates text chunks.
3. Text embeddings are generated using an AI model.
4. FAISS indexes the embeddings for efficient search.
5. User enters a question, and the app retrieves relevant content.
6. AI generates an answer based on the retrieved context.

## 🎯 Future Enhancements
- 📝 Support for multiple document formats (Word, Excel, etc.)
- 🔥 More AI models for improved responses
- 📊 Visualization of search results

## 💡 Contributing
Feel free to fork this repository, submit issues, or create pull requests to enhance the project!

## 📄 License
This project is licensed under the MIT License.

---

## 👤 Author
**Mitesh Nandan**

Data Analyst, AI/ML Enthusiast

### 🌐 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mitesh-nandan) [![Tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/mitesh.nandan) 
[![Kaggle](https://img.shields.io/badge/Kaggle-white?style=for-the-badge&logo=kaggle&logoColor=blue&color=f9f9f9)](https://www.kaggle.com/miteshnandan) [![Instagram](https://img.shields.io/badge/Instagram-E1306C?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/its.all.nostalgic/)

Let's collaborate and innovate together!

---

