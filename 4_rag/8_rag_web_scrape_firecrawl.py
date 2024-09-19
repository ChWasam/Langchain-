import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
# https://www.firecrawl.dev/app/api-keys
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl_dawn")


def create_vector_store():
    """Crawl the website, split the content, create embeddings, and persist the vector store."""
    # Define the Firecrawl API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    # Step 1: Crawl the website using FireCrawlLoader
    print("Begin crawling the website...")
    loader = FireCrawlLoader(
        api_key=api_key, url="https://www.dawn.com/news/1859742/taming-the-judiciary", mode="scrape")
    #  mode => crawl for whole website(you can easily blown all your tokens), scrape for specific page
    docs = loader.load()
    print("Finished crawling the website.")

    # Convert metadata values to strings if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

# --------------------------------------Explaination--------------------------------
    # doc.metadata.items(): Retrieves all key-value pairs from the document's metadata dictionary.
    # isinstance(value, list): Checks if the value is a list.
    # map(str, value):Applies the str function to each element in the list value.Ensures all elements are converted to strings (useful if the list contains non-string elements).
    # ", ".join(...):Concatenates the strings in the iterator, separating them with a comma and a space.Results in a single string with list elements separated by commas.
    # This code efficiently standardizes metadata values by converting lists into comma-separated strings. By ensuring that all metadata values are strings, it facilitates consistent data handling in subsequent processing steps, such as serialization, display, or input to language models. The use of isinstance and map(str, ...) provides robustness by handling different data types and avoiding errors.
# --------------------------------------------------------------------------------------




    # Step 2: Split the crawled content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")

    # Step 3: Create embeddings for the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 4: Create and persist the vector store with the embeddings
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(
        split_docs, embeddings, persist_directory=persistent_directory
    )
    print(f"--- Finished creating vector store in {persistent_directory} ---")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(
        f"Vector store {persistent_directory} already exists. No need to initialize.")

# Load the vector store with the embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


# Step 5: Query the vector store
def query_vector_store(query):
    """Query the vector store with the specified question."""
    # Create a retriever for querying the vector store
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    # Retrieve relevant documents based on the query
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# Define the user's question
query = "Do you know any age limit for judges in Pakistan?"

# Query the vector store with the user's question
query_vector_store(query)
