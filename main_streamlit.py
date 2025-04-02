import os
import time
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()
from aixplain.factories import ModelFactory

# Set up the page configuration and title
st.set_page_config(page_title="Address Parsing Chatbot", layout="wide")
st.title("My Address Parsing and Sorting Chatbot")

# Load environment variables

# Initialize the JSON parser and the language model
json_parser = JsonOutputParser()
selected_model = ModelFactory.get("6646261c6eb563165658bbb1")

# Define the minimum character count and validation function
MIN_CHARACTER_COUNT = 30
def is_valid_address(text: str) -> bool:
    if len(text) < MIN_CHARACTER_COUNT and not re.search(r'[,\d]', text):
        return False
    return True

# Use Streamlit's caching to load or create the vector store only once
@st.cache_resource(show_spinner=True)
def load_vector_store():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    persist_directory = os.path.join(os.getcwd(), "chroma_db_all")
    if not os.path.exists(persist_directory):
        st.info("Vector store not found. Creating a new one from 'output.txt'...")
        loader = TextLoader('output.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator="\n")
        docs = text_splitter.split_documents(documents)
        st.write(f"Number of document chunks: {len(docs)}")
        db_local = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        st.success("Vector store created!")
    else:
        st.info("Loading existing vector store...")
        db_local = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        st.success("Vector store loaded!")
    return db_local

# Load (or create) the vector store
db = load_vector_store()

# Create a text area for the user to input an address
query = st.text_area("Enter the address to parse:")

# When the user clicks the "Parse Address" button, process the input
if st.button("Parse Address"):
    start_time = time.time()
    
    if not is_valid_address(query):
        st.error("NOT A VALID ADDRESS. Please enter a valid address.")
    else:
        # Retrieve relevant documents from the vector store
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 20, "score_threshold": 0.3},
        )
        relevant_docs = retriever.invoke(query)
        
        # Build the two prompt inputs based on the retrieved documents and the query
        combined_input = (
            "Find me the region name and emirate name in the address with the help of the documents provided: "
            + query +
            "\n\nRelevant Documents:\n" +
            "\n\n".join([doc.page_content for doc in relevant_docs]) +
            "\n\nPlease provide an answer based only on the provided documents. Give me either both, region name and the emirate or if one is found, give me that and return Null for the other. The name need not match exactly. If there it looks similar go for it. If the address has part of the name in the document, go for it. Like 'Musaffah' instead of 'Al Musafah'.  Give me a dictionary in json format response. Also add corresponding region code and emirate code if available. If not available, return Null. If you are not sure, return Null. The keys should be 'region_name', 'region_code', 'emirate_name', 'emirate_code'. Only return the valid JSON. NO PREAMBLE"
        )
    
        combined_input_2 = (
            "Find me the addressee name, phone number or/and email, any instruction for the dilivery, villa number or flat number, PO Box number or code, building name or apartment name, street or/and landmark from the address: "
            + query +
            "\n\nReturn a dictionary in json with keys: addressee name (if available), phone number (if available), email (if available), delivery instructions (if available), villa number or flat number (if available), PO Box number or code (if available), floor number, building name or apartment name (if available) and street (if available), landmark (if available). If any of the information is not available, please return Null. Just give me the information without any preface. And return Null if you don't know. Only return the valid JSON. NO PREAMBLE"
        )
    
        # Run the model on both prompts
        with st.spinner("Parsing address..."):
            result_1 = selected_model.run({'text': combined_input})
            result_2 = selected_model.run({'text': combined_input_2})
    
        # Parse the JSON responses
        try:
            parsed_1 = json_parser.parse(result_1['data'])
        except Exception as e:
            st.error(f"Error parsing first result: {e}")
            parsed_1 = {}
    
        try:
            parsed_2 = json_parser.parse(result_2['data'])
        except Exception as e:
            st.error(f"Error parsing second result: {e}")
            parsed_2 = {}
    
        # Combine the two dictionaries (keys from the second prompt override if overlapping)
        final_dict = {**parsed_2, **parsed_1}
    
        # Display the final parsed output as JSON
        st.subheader("Parsed Output:")
        st.json(final_dict)
    
    end_time = time.time()
    st.write(f"Processing completed in {end_time - start_time:.4f} seconds")
