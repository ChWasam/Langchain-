from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_google_genai  import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                client_options=None,
                                transport=None,
                                additional_headers=None,
                                client=None,
                                async_client=None)

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

#  As we will become more practical we can make api calls 
# other examples of RunnableLambda are
# 1. RunnableLambda(lambda x: x.upper())
# 2. RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# len(x.split()):
# x.split() splits the string x into a list of words by default (it splits by whitespace unless
# specified otherwise).
# len(x.split()) calculates the number of elements in the list, which is effectively the word 
# count in the string x.


# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)
