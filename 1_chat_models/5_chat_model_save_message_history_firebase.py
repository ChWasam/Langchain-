# Example Source: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/
#  Here we will start saving our massages to the cloud using Google Firestore.
#  We will use the langchain_google_firestore package to save our chat history.
from dotenv import load_dotenv
from google.cloud import firestore  # type: ignore
# Also add "poetry add google-cloud-firestore-stubs" to the pyproject.toml file
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage

#  Explore other options from lanchain docs. All you need to do is to store messages 
#  We have Fire base as a google cloud and we want our local computer to communicate with the cloud.
"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
#  We have Fire base as a google cloud and we want our local computer to communicate with the cloud.Hence we need to install the google cloud sdk
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account to access the Firebase project 
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation

Details in the README.md file of this folder.
"""

load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchain-demo-82f9f"
SESSION_ID = "user_session_new1"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    client_options=None,
    transport=None,
    additional_headers=None,
    client=None,
    async_client=None,
)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    ai_message = AIMessage(content=ai_response.content)
    chat_history.add_ai_message(ai_message)

    print(f"AI: {ai_response.content}")
