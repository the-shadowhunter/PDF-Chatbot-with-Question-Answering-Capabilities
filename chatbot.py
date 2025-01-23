import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Store the chat history globally
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error processing file {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Answer questions using the provided context or the chat history. 
    If the answer is not available in the context, reply:
    "Answer is not available in the context. Do you want me to search the internet or access the resources to help you?"
    Context: {context} \n
    Previous Conversation: {history} \n
    Current Question: {question} \n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def search_internet(query):
    import requests
    endpoint = f"https://api.searchengine.com/v1/search?q={query}&key={API_KEY}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        result = response.json()
        return result.get("answer", "No relevant information found.")
    else:
        return "Error fetching data from the internet."

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        st.error("FAISS index not found. Please upload and process PDF files first.")
        return
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Get the chat history
    chat_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
    
    chain = get_conversational_chain()
    
    # Generate the response
    response = chain({
        "input_documents": docs,
        "history": chat_history,
        "question": user_question
    }, return_only_outputs=True)
    answer = response["output_text"]
    #If the answer is not avialable in the context
    if "Answer is not avialabe in the context." in answer:
        st.write(answer)
        user_choice = st.radio(
            "Do you want me to search the internet or access other resources?",
            ("Yes", "No"),
            index = 1
        )
        user_choice = user_choice.lower()
        if user_choice == "yes":
            st.write("Searching the internet ...")
            external_answer = search_internet(user_question)
            st.write("Reply from the external resource: ", external_answer)
        else:
            st.write("Okay, sticking to the local context.")
    else:
        # Add the new question and answer to the chat history
        st.session_state.chat_history.append((user_question, response["output_text"]))
        
        # Display the response
        st.markdown(f"**Reply:** {response['output_text']}")


def main():
    # Set up the page with styled elements
    st.set_page_config(page_title="Chat Bot", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
            color: #333;
        }
        [data-testid="stTextInputRootElement"] {
            background-color: #fffff;
            color: white;
            border: 2px solid #007BFF;
            border-radius: 4px;
            padding: 0.5em 1em;
            margin-bottom: 1em;
            font-size: 1em;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5em 1em;
            cursor: pointer;
            font-size: 1em;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .button-container {
            display: flex;
            justify-content: space-between;
            gap: 1em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Chat with PDF using the Chat Bot")
    
    with st.form("question_form"):
        user_question = st.text_input("Ask a Question from the PDF files")
        submit_button = st.form_submit_button("Enter")
        
    if user_question and submit_button:
        user_input(user_question)

    # with st.form("question_form"):
    #     # Input field for the user question
    #     user_question = st.text_input(
    #         "Ask a Question from the PDF files",
    #         value=st.session_state.get("user_question", "")
    #     )
        
    #     # Place buttons side by side
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         clear_button = st.form_submit_button("Clear")
    #     with col2:
    #         submit_button = st.form_submit_button("Enter")
        
    #     # Handle button actions
    #     if clear_button:
    #         st.session_state["user_question"] = ""  # Clear the input
    #     elif submit_button and user_question:
    #         st.session_state["user_question"] = user_question
    #         user_input(user_question)  # Process the input


    # Chat History Display
    st.write("### Chat History")
    if st.session_state.get("chat_history"):
        total_questions = len(st.session_state.chat_history)
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            reverse_index = total_questions - i + 1  # Calculate reverse index
            st.markdown(f"**Q{reverse_index}:** {q}")
            st.markdown(f"**A{reverse_index}:** {a}")
            st.markdown("---")  # Adds a horizontal divider
    else:
        st.write("No chat history yet.")

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                    else:
                        st.error("No text extracted from the uploaded PDFs.")
            else:
                st.error("Please upload PDF files before clicking submit.")

if __name__ == "__main__":
    main()

