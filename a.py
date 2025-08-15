import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
import requests
import urllib3
from dotenv import load_dotenv
import docx
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import zipfile
import rarfile
import tempfile
import mimetypes
from io import BytesIO, StringIO

# ---------------------------
# Fix Torch watcher issue for Streamlit
# ---------------------------
os.environ["STREAMLIT_WATCHER_IGNORE_TORCH"] = "1"

# ---------------------------
# Suppress SSL warnings
# ---------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error("⚠️ Anthropic API key not found in .env")
    st.stop()

# ---------------------------
# Configuration for file uploads
# ---------------------------
# Set max file size to 1GB (adjust as needed)
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB in bytes

# Supported file types
SUPPORTED_EXTENSIONS = {
    'pdf': 'PDF files',
    'txt': 'Text files',
    'docx': 'Word documents',
    'doc': 'Word documents (older format)',
    'eml': 'Email files',
    'msg': 'Outlook email files',
    'mbox': 'Mailbox files',
    'csv': 'CSV files',
    'html': 'HTML files',
    'htm': 'HTML files',
    'xml': 'XML files',
    'json': 'JSON files',
    'md': 'Markdown files',
    'rtf': 'Rich Text Format',
    'zip': 'ZIP archives',
    'rar': 'RAR archives'
}

# ---------------------------
# Custom LangChain LLM wrapper for Claude API
# ---------------------------
class AnthropicChatLLM(LLM):
    model_name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1000
    temperature: float = 0.3
    api_key: str = ANTHROPIC_API_KEY

    @property
    def _llm_type(self):
        return "anthropic_chat"

    def _call(self, prompt: str, stop=None) -> str:
        headers = {
            "x-api-key": self.api_key.strip(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        data = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30,
                verify=True
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "content" in result and len(result["content"]) > 0:
                return result["content"][0]["text"]
            else:
                return "No response content received"
                
        except requests.exceptions.SSLError:
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=30,
                    verify=False
                )
                response.raise_for_status()
                result = response.json()
                
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0]["text"]
                else:
                    return "No response content received"
            except Exception as e:
                return f"❌ SSL Error calling Anthropic API: {e}"
                
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return "❌ Authentication failed. Please check your Anthropic API key."
            elif response.status_code == 429:
                return "❌ Rate limit exceeded. Please try again later."
            else:
                return f"❌ HTTP Error {response.status_code}: {e}"
        except Exception as e:
            return f"❌ Error calling Anthropic API: {e}"

# Initialize LLM
llm = AnthropicChatLLM()

# ---------------------------
# Local embedding model
# ---------------------------
try:
    local_embedding_model = SentenceTransformer(
        r"C:\Users\v-priya.tukkannavar\Downloads\RAG1\RAG\all-MiniLM-L6-v2"
    )
except Exception:
    st.warning("Local model path not found. Downloading model...")
    local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return local_embedding_model.encode(texts).tolist()
    def embed_query(self, text):
        return local_embedding_model.encode([text])[0].tolist()

embeddings = LocalEmbeddings()

# ---------------------------
# File processing functions
# ---------------------------
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {file.name}: {e}")
        return ""

def extract_text_from_email(file):
    """Extract text from email files (.eml, .msg)"""
    try:
        file_content = file.read()
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8', errors='ignore')
        
        # Parse email
        msg = email.message_from_string(file_content)
        
        text = f"Subject: {msg.get('Subject', 'No Subject')}\n"
        text += f"From: {msg.get('From', 'Unknown')}\n"
        text += f"To: {msg.get('To', 'Unknown')}\n"
        text += f"Date: {msg.get('Date', 'Unknown')}\n\n"
        
        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True)
                    if body:
                        text += body.decode('utf-8', errors='ignore')
                elif part.get_content_type() == "text/html":
                    # Simple HTML stripping (you might want to use BeautifulSoup for better parsing)
                    body = part.get_payload(decode=True)
                    if body:
                        html_content = body.decode('utf-8', errors='ignore')
                        # Basic HTML tag removal
                        import re
                        clean_text = re.sub(r'<[^>]+>', '', html_content)
                        text += clean_text
        else:
            body = msg.get_payload(decode=True)
            if body:
                text += body.decode('utf-8', errors='ignore')
        
        return text
    except Exception as e:
        st.error(f"Error reading email {file.name}: {e}")
        return ""

def extract_text_from_archive(file):
    """Extract text from ZIP/RAR archives"""
    text = ""
    try:
        file_extension = file.name.lower().split('.')[-1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        if file_extension == 'zip':
            with zipfile.ZipFile(temp_file_path, 'r') as archive:
                for file_info in archive.filelist:
                    if not file_info.is_dir():
                        try:
                            with archive.open(file_info) as archived_file:
                                # Create a file-like object from the archived file
                                archived_content = archived_file.read()
                                
                                # Try to process based on file extension
                                archived_filename = file_info.filename.lower()
                                if archived_filename.endswith('.txt'):
                                    text += archived_content.decode('utf-8', errors='ignore') + "\n"
                                elif archived_filename.endswith('.pdf'):
                                    # For PDFs in archives, you'd need more complex handling
                                    pass
                        except Exception as e:
                            st.warning(f"Could not extract {file_info.filename}: {e}")
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
    except Exception as e:
        st.error(f"Error processing archive {file.name}: {e}")
    
    return text

def extract_text_from_file(file):
    """Main function to extract text from any supported file type"""
    file_extension = file.name.lower().split('.')[-1]
    
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        st.error(f"File {file.name} is too large ({file.size / (1024*1024):.1f}MB). Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB")
        return ""
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension in ['docx']:
        return extract_text_from_docx(file)
    elif file_extension in ['eml', 'msg']:
        return extract_text_from_email(file)
    elif file_extension in ['txt', 'md', 'csv', 'json', 'xml', 'html', 'htm', 'rtf']:
        try:
            content = file.read()
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='ignore')
            return content
        except Exception as e:
            st.error(f"Error reading text file {file.name}: {e}")
            return ""
    elif file_extension in ['zip', 'rar']:
        return extract_text_from_archive(file)
    else:
        st.warning(f"Unsupported file type: {file_extension}")
        return ""

def get_files_text(uploaded_files):
    """Extract text from multiple uploaded files"""
    all_text = ""
    processed_files = []
    
    for file in uploaded_files:
        st.info(f"Processing: {file.name}")
        text = extract_text_from_file(file)
        if text.strip():
            all_text += f"\n\n--- Content from {file.name} ---\n\n{text}"
            processed_files.append(file.name)
        else:
            st.warning(f"No text extracted from {file.name}")
    
    if processed_files:
        st.success(f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}")
    
    return all_text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create FAISS vector store"""
    if not text_chunks:
        st.error("No text chunks to process")
        return
    
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# ---------------------------
# Conversational chain
# ---------------------------
def get_conversational_chain():
    prompt_template = """
You are a helpful assistant that answers questions based on the provided context from various document types including PDFs, emails, and other files.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based only on the information provided in the context above.
2. If the answer is not available in the context, clearly state "The answer is not available in the provided context."
3. Provide a clear, concise, and helpful response.
4. If relevant, quote specific parts from the context to support your answer.
5. If the context contains information from multiple files, mention which file the information comes from when relevant.

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=llm, prompt=prompt)

def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ No FAISS index found. Please upload and process files first. Error: {e}")
        return

    try:
        docs = new_db.similarity_search(user_question, k=5)
        if not docs:
            st.warning("No relevant documents found for your question.")
            return
            
        docs_text = "\n\n".join([d.page_content for d in docs])

        chain = get_conversational_chain()
        
        with st.spinner("Generating response..."):
            response = chain.invoke({"context": docs_text, "question": user_question})
            
        if isinstance(response, dict) and 'text' in response:
            st.write("💡 **Reply:**", response['text'])
        else:
            st.write("💡 **Reply:**", str(response))
            
    except Exception as e:
        st.error(f"Error processing your question: {e}")

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="Multi-File Chat with Claude", page_icon="📁", layout="wide")
    st.header("📁 Chat with Multiple File Types (Claude AI)")
    
    # Display supported file types
    st.markdown(f"""
    **Supported file types:** {', '.join(SUPPORTED_EXTENSIONS.keys())}
    
    Upload multiple files including PDFs, Word documents, emails, text files, and archives. The app uses:
    - **Local embeddings**: SentenceTransformers for document processing  
    - **Claude API**: Anthropic's Claude for intelligent responses
    - **Maximum file size**: {MAX_FILE_SIZE / (1024*1024):.0f}MB per file
    """)

    # Sidebar upload
    with st.sidebar:
        st.title("📂 File Upload")
        st.markdown("Upload your files to get started:")
        
        # Create list of supported extensions for file uploader
        supported_types = list(SUPPORTED_EXTENSIONS.keys())
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            accept_multiple_files=True,
            type=supported_types,
            help=f"Supported: {', '.join(supported_types)}"
        )
        
        if st.button("Submit & Process", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                with st.spinner("Processing files..."):
                    try:
                        raw_text = get_files_text(uploaded_files)
                        if not raw_text.strip():
                            st.error("No text found in the uploaded files.")
                            return
                        
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("No text chunks created from files.")
                            return
                            
                        get_vector_store(text_chunks)
                        st.success("✅ File processing complete!")
                        st.info(f"Processed {len(text_chunks)} text chunks from {len(uploaded_files)} file(s)")
                        
                    except Exception as e:
                        st.error(f"Error processing files: {e}")
        
        # Show current status
        if os.path.exists("faiss_index"):
            st.success("📊 Vector database ready")
        else:
            st.info("📊 No vector database found")
        
        # File type legend
        st.markdown("### 📋 Supported File Types")
        for ext, desc in SUPPORTED_EXTENSIONS.items():
            st.text(f"• .{ext} - {desc}")

    # Main question input area
    st.subheader("Ask Questions")
    user_question = st.text_input(
        "Enter your question about the uploaded files:",
        placeholder="e.g., What is discussed in the documents? Summarize the main points from the emails."
    )
    
    if user_question:
        user_input(user_question)
    
    # Add example questions
    if os.path.exists("faiss_index"):
        st.subheader("💡 Try these example questions:")
        example_questions = [
            "What is the main topic across all documents?",
            "Can you provide a summary of all uploaded content?",
            "What are the key points mentioned in the emails?",
            "Are there any specific recommendations or action items?",
            "What files were uploaded and what do they contain?",
            "Compare information across different documents"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            col = cols[i % 2]
            if col.button(question, key=f"example_{i}"):
                user_input(question)

if __name__ == "__main__":
    main()