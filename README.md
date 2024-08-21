# Python WhatsApp API Bot 
## Purpose
This is a Flask application which enables us to receive and send WhatsApp messages through the WhatsApp API. We generate responses to user questions with RAG, i.e., we retrieve relevant information from a vector database and then use an LLM to generate a response based on the question and the relevant information. 

## High-level functionality
When the WhatsApp API sends us a message (e.g., a user sent a message to our number), we receive this message (HTTP POST) in the `app.views.py` file. For incoming WhatsApp messages, the `process_whatsapp_messages()` method is called (this method is defined in the `app.utils.whatsapp_utils.py` file). In the `app.utils.whatsapp_utils.py` file, the WhatsApp message from the API is preprocessed and then the `generate_response()` method is called (defined in the `app.utils.langchain_bot.py` file). In the `app.utils.langchain_bot.py` file, we mostly use the LangChain framework with OpenAI's LLM and embedding models and Pinecone to generate a response message with RAG (retrieval-augmented generation).
