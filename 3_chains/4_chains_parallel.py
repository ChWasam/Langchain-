from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                client_options=None,
                                transport=None,
                                additional_headers=None,
                                client=None,
                                async_client=None)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)
cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)
#  Previously when you will run pro chain it takes 30 s and then cons chain will run for 30 s
#  But now using parallel chain it will run both chains at the same time and will take 30 s
# Hence reducing the time and are more efficient
#  For Applications require speedy output we can use parallel chains

# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Output
print(result)


# ------------------------------ New concept --------------------------------

# RunnableLambda(lambda x: print("final output", x) or combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
 

# Ah, I see! Let's clarify how both sides of the or are giving results in this case.

# The key point here is understanding how 'or' operator works in Python:

# The 'or' operator: It evaluates the left-hand side first. If it's truthy (i.e., not None, False, 0, or an empty value), the left-hand side is returned, and the right-hand side is not evaluated. However, if the left-hand side is falsy (like None, which is the case for the print() function), then Python proceeds to evaluate and return the right-hand side.
# Why both sides are evaluated here:
# print("final output", x): The print() function executes and prints "final output" along with the value of x, but it always returns None.
# combine_pros_cons(...): Since print() returns None, Python evaluates the right-hand side of the or statement, i.e., combine_pros_cons(...), and returns its result.
# Why both sides appear to give results:
# The left-hand side (print(...)) gives a result because it is evaluated, and the side effect of printing happens. Even though print() returns None, it still performs the action of outputting text to the console.

# The right-hand side (combine_pros_cons(...)) is evaluated because the or operator needs to determine a truthy value. Since print(...) returns None, Python evaluates and returns the result of combine_pros_cons(...).

