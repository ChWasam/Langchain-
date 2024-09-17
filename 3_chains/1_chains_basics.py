# Chains Used for chat automation 
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                client_options=None,
                                transport=None,
                                additional_headers=None,
                                client=None,
                                async_client=None)

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()

#  What StrOutputParser() does is basically like every time we get the result we do .content. That is exactly string parser does. 
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)
