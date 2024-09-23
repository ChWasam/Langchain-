from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import BaseSingleActionAgent, BaseMultiActionAgent
from dotenv import load_dotenv
import typing

# Load environment variables from .env file
load_dotenv()

# Define Tools
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary # type: ignore
    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."


# Define the tools that the agent can use
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a ChatOpenAI model
llm: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")

# Create a structured Chat Agent with Conversation Buffer Memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Initialize agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=typing.cast(typing.Union[BaseSingleActionAgent, BaseMultiActionAgent], agent),
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    try:
        # Invoke the agent with the user input and the current chat history
        response = agent_executor.invoke({"input": user_input})
        print("Bot:", response["output"])

        # Add the agent's response to the conversation memory
        memory.chat_memory.add_message(AIMessage(content=response["output"]))
    except ValueError as e:
        print(f"An error occurred: {e}")


#  If we will add Human Message we will get the following error
# An error occurred: Unexpected message with type <class 'langchain_core.messages.system.SystemMessage'> at the position 1.

#  The SystemMessage is removed to avoid message type conflicts. You can instead ensure the agent’s initial_message is passed via the prompt template directly.

# # Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
# initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
# memory.chat_memory.add_message(SystemMessage(content=initial_message))