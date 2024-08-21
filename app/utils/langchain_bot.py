import os #import the os module to access the environment variables later
from dotenv import load_dotenv #import the load_dotenv function from the dotenv module to load the environment variables
#initialize the environment
load_dotenv() 


#Initialize OpenAI 
openai_api_key = os.getenv("OPENAI_API_KEY")
# Define/ assign the embeddings model with the OpenAIEmbeddings class
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=openai_api_key
)
# Define/ assign the language model with the ChatOpenAI class
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4-turbo")


#initializing pinecone
from langchain_pinecone import PineconeVectorStore
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "isaac" #name of the index
name_space = "first_try" #name of the namespace
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding,namespace=name_space, pinecone_api_key=pinecone_api_key, text_key = "transcript")
#create a retriever that is used for retrieving relevant embeddings from the Pinecone vector database via similarity search 
retriever = vectorstore.as_retriever()


#Create a history aware retriever chain – a chain that first reforumaltes the user's question based on the chat history and then retrieves relevant embeddings from the Pinecone vector database
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Define the system prompt, which tells the model to only reformulate the initial query based on the chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
# Define the prompt for the history aware retriever, i.e. the template that we use to combine the system prompt with the chat history and the user input
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Define the history aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


#create the qa chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Define the system prompt for the QA chain, i.e. the instructions that we give to the llm telling the llm how it should act (e.g. from isaac's perspective)
qa_system_prompt = """You are Isaac, who is an influencer. You act like you are him. \
Isaac usually does reels where he interviews people and asks them about recommendations. \
When you give recommendations, please always mention the instagram_url and the person who made the recommendation for every single recommendation you make.\
Please don't format it like a list, but rather like a conversational response, pretty casual. \

How you can find the relevant information: 
You can see relevant transcripts, usually include recommendations, with the corresponding instagram url's  and captions (the person who made the recommendation is usually tagged) right after each transcript!\
Only use information provided in the following, otherwise, just say you don't know \

{context}"""
# Define the prompt for the QA chain, i.e. the template that we use to combine the system prompt with the chat history and the user input
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
from langchain.prompts import PromptTemplate
# Define which information from each retrieved document should be included in the context and how it should be formatted
document_prompt = PromptTemplate(
    input_variables=["page_content", "instagram_url", "caption"],
    template="Transcript: {page_content}\n Instagram URL: {instagram_url} \n Caption: {caption}"
)


# Create the two final chains
# The first chain is the question answering chain, which is responsible for answering the user's question
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt = document_prompt)
# The second chain combines the history aware retriever with the question answering chain, by first retrieving relevant embeddings from the Pinecone vector database and then passing them to the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Define how user histories are stored
from langchain_community.chat_message_histories import RedisChatMessageHistory
#Create a Redis History
history = RedisChatMessageHistory("foo", url="redis://red-cqp3af88fa8c73c5s8v0:6379")

#A random import I don't get
from typing import Optional


from langchain_core.runnables.history import RunnableWithMessageHistory
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://red-cqp3af88fa8c73c5s8v0:6379"
    ),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Define the function that is called when a user sends a message via WhatsApp, i.e. generate the response
def generate_response(message_body, wa_id, name):
    # Retrieve the chat history of the user
    response = conversational_rag_chain.invoke(
    {"input": message_body},
    config={
        "configurable": {"session_id": wa_id}
    },  
    )["answer"]
    return response
