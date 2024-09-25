import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Who is Odysseus' wife?"
# -----------------------------------------------------
# query = "Who is wasam?" 
# UserWarning: No relevant docs were retrieved using the relevance score threshold 0.9

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
    #  Heigher the score_threshold, the more similar the documents will be to the query
    # "k": 3=> What it's going to do is it's going to return the top 3 most relevant documents
)
#  In RAG if you are not getting results you might be too strict with your score_threshold

relevant_docs = retriever.invoke(query)
# relevant_docs is list

# Display the relevant results with metadata
#  All we are doing in this code is we are just printing the relevant documents
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")



#  The enumerate() function in Python allows you to loop over an iterable (like a list, tuple, or string) and keep track of both the index and the element at the same time. This is particularly useful when you want to know the position of an element as you iterate through it.

# In a for loop, when you write< for i, doc in enumerate(relevant_docs, 1)> , you're using tuple unpacking. The enumerate() function returns a tuple for each iteration where:
# The first element of the tuple is the index (i).
# The second element of the tuple is the current document object (doc).
#  More detail on notion
