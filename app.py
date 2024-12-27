import streamlit as st
import os
import logging
from PIL import Image
import tensorflow as tf
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import ollama

# Configure TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.debugging.set_log_device_placement(True)
logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Constants
DOC_PATH = "./medicinal-herbs.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"
CLASSIFICATION_MODEL_PATH = "./medicinal-herb-final-model.h5"
INPUT_IMAGE_SIZE = (150, 150)  # Model-specific input size

CLASS_TO_PLANT_NAME = {0: 'Aloevera', 1: 'Bamboo', 2: 'Castor', 3: 'Neem', 4: 'Tamarind'}

# Streamlit Page Configuration
st.set_page_config(page_title="RAG Based Herb Bot", page_icon="🌿", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #EAF9FF; padding-top: 20px; }
    .title { text-align: center; font-size: 40px !important; color: #0078D4; font-weight: bold; }
    .subtitle { text-align: left; color: #0078D4; font-size: 20px; }
    .bot-response { background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #0078D4; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

def load_classification_model():
    """Load the plant classification model."""
    return tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)


def classify_plant(image, model):
    """
    Classify the plant from an uploaded image.

    Args:
        image: PIL Image - The input image to classify.
        model: tf.keras.Model - The pre-trained model used for classification.

    Returns:
        str: Predicted plant name.
    """
    target_size = (150, 150)  # Adjusted input size to smaller dimensions
    image = image.resize(target_size)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0  # Normalize

    predictions = model.predict(img_array)
    plant_class = tf.argmax(predictions, axis=-1).numpy()[0]
    return CLASS_TO_PLANT_NAME.get(plant_class, "Unknown Plant")


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db

def create_chain(vector_db, llm):
    """Create the RAG QA chain."""
    retriever = vector_db.as_retriever()

    template = """Answer based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    logging.info("Chain created.")
    return chain


def main():
    st.markdown('<p class="title">🌿 RAG Based Herb Chatbot</p>', unsafe_allow_html=True)
    classification_model = load_classification_model()

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_image:
        image = Image.open(uploaded_image)
        plant_class = classify_plant(image, classification_model)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=False)
        st.markdown(f'<p class="subtitle">Predicted Herb: {plant_class}</p>', unsafe_allow_html=True)

        # user_input = st.text_input("Enter your question about {plant_class}", "")
        user_input = st.text_input(f"Enter your question about {plant_class}", "")

        if user_input:
            with st.spinner("Processing..."):
                try:
                    llm = ChatOllama(model=MODEL_NAME)
                    vector_db = load_vector_db()
                    if vector_db is None:
                        st.error("Failed to load or create the vector database.")
                        return

                    chain = create_chain(vector_db, llm)
                    query = f"Herb: {plant_class}. {user_input}"
                    response = chain.invoke({"query": query})

                    st.markdown("**Assistant:**")
                    st.write(response['result'])
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()