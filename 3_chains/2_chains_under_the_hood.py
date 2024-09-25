from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

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

#  Chains the manual way / chains in the background
# Create individual runnables (steps in the chain)
# Think of runable as a task 
# Lamda  is a function that is defined without a name.
# RunnableLambda is a task that is defined without a name.
# lambda is a function her  that takes x as input and returns prompt_template.format_prompt(**x)
# The format_prompt(**x) method call means that the x argument is being unpacked into keyword arguments. The double asterisk (**) operator unpacks the dictionary x, allowing you to pass its key-value pairs as keyword arguments to the format_prompt method.
#  Keyward Arguments 
#  greet(age=28, name="Charlie")  # age and name are keyword arguments
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
# Sequence of Runnables, where the output of each is the input of the next.
chain: RunnableSequence = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

#  First is a single runable, middle is a list of runables, last is a single runable 

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(response)
