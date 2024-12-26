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
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.debugging.set_log_device_placement(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #Disable GPU
# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./medicinal-herbs.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"
CLASSIFICATION_MODEL_PATH = "./model.h5" 

# Map class indices to plant names
CLASS_TO_PLANT_NAME = {
    0: "Aloevera",
    1: "Neem",
    2: "Tamarind"
}

# Set page configuration
st.set_page_config(page_title="RAG Based Herb Bot", page_icon="🌿", layout="centered")

st.write(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('GPU'))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #EAF9FF;
        padding-top: 20px;
    }
    .title {
        text-align: center;
        font-size: 30px;
        color: #0078D4;
        font-weight: bold;
        margin-top: 0;
    }
    .subtitle {
        text-align: center;
        color: #0078D4;
        font-size: 18px;
    }
    .bot-response {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0078D4;
        margin-bottom: 10px;
    }
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
    # Resize the image to match the model's expected input size
    target_size = (255, 255)  # Adjust based on your model's input size
    image = image.resize(target_size)

    # Convert image to array and normalize it if required
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Debugging information
    # st.write("Input shape after preprocessing:", img_array.shape)
    # st.write("Model expected input shape:", model.input_shape)

    # Make predictions
    predictions = model.predict(img_array)
    plant_class = tf.argmax(predictions, axis=-1).numpy()[0]  # Get class index

    # Map the class index to the plant name
    plant_name = CLASS_TO_PLANT_NAME.get(plant_class, "Unknown Plant")
    return plant_name

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


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    # Main UI layout
    st.markdown('<p class="title">🌿 RAG Based Herb Chatbot</p>', unsafe_allow_html=True)

    # Load the classification model
    classification_model = load_classification_model()

    # File uploader
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    # Image and text inputs
    if uploaded_image:
     image = Image.open(uploaded_image)
     plant_class = classify_plant(image, classification_model)
     st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
     st.markdown(f'<p class="subtitle">Medicinal Herb Predicted: {plant_class}</p>', unsafe_allow_html=True)
     # Plant Image Input
     user_input = st.text_input("Enter your question about the herb:", "")
    else:
     st.info("Please upload a plant image to get started.")
    
     
    

    if uploaded_image and user_input:
        with st.spinner("Processing..."):
            try:
                # Step 1: Classify the plant
                # image = Image.open(uploaded_image)
                # plant_class = classify_plant(image, classification_model)
                # st.success(f"Plant identified as: {plant_class}")

                # Step 2: Initialize the language model
                llm = ChatOllama(model=MODEL_NAME)

                # Step 3: Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Step 4: Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Step 5: Create the chain
                chain = create_chain(retriever, llm)

                # Step 6: Generate response
                query = f"Plant: {plant_class}. {user_input}"
                response = chain.invoke(input=query)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    # else:
    #     st.info("Please upload a plant image and enter a question to get started.")


if __name__ == "__main__":
    main()


