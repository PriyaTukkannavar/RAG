import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from langchain_anthropic import ChatAnthropic

# Fix Torch watcher issue for Streamlit
os.environ["STREAMLIT_WATCHER_IGNORE_TORCH"] = "1"

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    st.error("⚠️ Anthropic API key not found. Please set ANTHROPIC_API_KEY in .env")
    st.stop()

# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0.3,
    api_key=ANTHROPIC_API_KEY
)

# Load local embedding model
local_embedding_model = SentenceTransformer(
    r"C:\Users\PRIYA\Downloads\model\all-MiniLM-L6-v2"
)

# LangChain wrapper for SentenceTransformer
class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return local_embedding_model.encode(texts).tolist()
    def embed_query(self, text):
        return local_embedding_model.encode([text])[0].tolist()

embeddings = LocalEmbeddings()

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Create FAISS vector store
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create conversational chain with Claude
def get_conversational_chain():
    prompt_template = """
    Answer the question from the context below.
    If the answer is not in the context, say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=llm, prompt=prompt)

# Handle user question
def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ No FAISS index found. Please upload and process PDFs first. Error: {e}")
        return

    docs = new_db.similarity_search(user_question, k=5)
    docs_text = "\n\n".join([d.page_content for d in docs])

    chain = get_conversational_chain()
    response = chain.run(context=docs_text, question=user_question)
    st.write("💡 Reply:", response)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF with Claude", page_icon="💬", layout="wide")
    st.header("💬 Chat with your PDF (Offline Embeddings + Claude LLM)")

    # Sidebar upload
    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text found in the uploaded PDFs.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDF processing complete!")

    # Question input
    user_question = st.text_input("Ask a question from the uploaded PDFs")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
