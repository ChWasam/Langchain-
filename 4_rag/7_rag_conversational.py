# We will generate every result based on the conversation history and the information from vector store

import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatGoogleGenerativeAI model
llm:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    client_options=None,
    transport=None,
    additional_headers=None,
    client=None,
    async_client=None,
)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
# Simply you are getting the question just rephrase it for the vector store so that we can retrieve the proper information

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        # A placeholder which can be used to pass in a list of messages.
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
#  The retriever used here will also bring relevant information from the vector store
# It uses the LLM to reformulate the user’s question based on the chat history, making sure that any references to past messages are resolved.
# The retriever then searches for relevant documents or past conversation snippets based on this contextualized question.
# This ensures that the LLM can answer the question using both retrieved information and its own capabilities.


# Does retriver will bring back input related content ?
# We are going to use the history_aware_retriever to get the information from the vector store


# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

#  According to my understanding the context is linked to history_aware_retriever through create_retrieval_chain => create_retrieval_chain(history_aware_retriever, question_answer_chain)and 


# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        # A placeholder which can be used to pass in a list of messages.
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
#  It is known as history_aware_retriever because based on the "conversation history + current input )it will be able to retrieve the information from the vector store
# history_aware_retriever => pulling information from vector store 
# question_answer_chain => Taking in the user input and actually responding answers to users and it all has context(information from vectorstore and conversation history) 


#  <---------------------Complete story------------------------>
# All we are going to do is set up this retrieval chain 
# ALL this is going to do is we will be able to 
# 1. retrieve information from our vector store => history_aware_retriever
# 2. we want to have awareness of all the conversation we have uptil this point => question_answer_chain
# Based on information from the vector store and the conversation history, we will be able to generate a result
# FEW STEPS REQUIRED TO SET THIS UP
#   we need to be able to grab information from (vectorstore and conversation history)
#   We will be using create_stuff_documents_chain, which will take in the <(bunch of documents)related to current query  + (what is going on)>  and pass/feed them into the LLM 
#  We also need to give to the LLM  what is going on (conversation) so the LLM is aware of what it needs to do  
#  The above part will help in responding to questions
#  But where does this (document and conversation) come from ? 
#  For that we need to understand the history_aware_retriever
#  This is where we will start  working with a vector store 
# llm => doing the thinking and generating the response
#  retriever => to retrieve information from the vector store 
# contextualize_q_system_prompt => provides context for what's going on
# up to you to reformulate the question so that we can properly search for information outside vector store 
# Simply you are getting the question just rephrase it for the vector store so that we can retrieve the proper information

#  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
