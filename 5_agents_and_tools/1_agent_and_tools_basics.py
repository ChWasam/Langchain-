import typing
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import BaseSingleActionAgent, BaseMultiActionAgent

# Load environment variables from .env file
load_dotenv()


# Define a very simple tool function that returns the current time
#  We have these arguments to avout mess up if we have any input but we don't use them 
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


# List of tools available to the agent
tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the current time",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
# As agent is nothing more than an LLM that has  been provided a specific prompt to guide its behavior.
# Use the following promp template to perform actions.
prompt = hub.pull("hwchase17/react")

# Initialize a ChatOpenAI model
llm:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    client_options=None,
    transport=None,
    additional_headers=None,
    client=None,
    async_client=None,
)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
#  Agent executor is just going to manage the run of an agent as it goes of to solve problems 
# Allowing us to get tools 
#  Executing the run and putting information back and forth
#  No need to dive in to it ,The main thing is Whenever i need to run my agent i need to use an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=typing.cast(BaseSingleActionAgent | BaseMultiActionAgent, agent),
    tools=tools,
    verbose=True,
)

"By using typing.cast(BaseSingleActionAgent | BaseMultiActionAgent, agent), you are explicitly telling the type checker that even though the agent is currently typed as Runnable[Any, Any], you are confident it should be treated as a BaseSingleActionAgent or BaseMultiActionAgent."
"Why This Works:"
"The typing.cast tells the type checker, “I know this might look like a Runnable (or some other type), but treat it as either BaseSingleActionAgent or BaseMultiActionAgent."
"This removes the error because you’ve made it explicit that you want the type of the agent to fit the expected types."

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

#  At the end it returns an object that contains the input and the response as output

# Print the response from the agent
print("response:", response)
