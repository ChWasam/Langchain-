from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]


# ---- LangChain OpenAI Chat Model Example ----

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----

# Create an Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model1 = ChatAnthropic(model_name="claude-3-haiku-20240307",timeout=None,api_key=None)

result1 = model1.invoke(messages)
print(f"Answer from Anthropic: {result1.content}")
# ('Your credit balance is too low to access the Claude API. Please go to Plans & Billing to upgrade or purchase credits.)


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model2:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    client_options=None,
    transport=None,
    additional_headers=None,
    client=None,
    async_client=None,
)

result2 = model2.invoke(messages)
print(f"Answer from Google: {result2.content}")
