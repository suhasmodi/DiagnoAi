from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_chroma.vectorstores import Chroma
import logging

logger = logging.getLogger(__name__)

class MedicalAssistant:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY', '2ca8a972b31040a97bf1b60f94c04aef876649d0467f7b46021dd0bb56d5792c')
        
        # Initialize the embeddings
        self.embeddings = GPT4AllEmbeddings()
        
        # Set up the Chroma vector store
        self.chroma_db_path = "chroma_db2"
        self.retriever = Chroma(
            persist_directory=self.chroma_db_path, 
            embedding_function=self.embeddings
        ).as_retriever(search_kwargs={"k": 5})
        
        # Create LLM instance
        self.llm_model = ChatTogether(
            together_api_key=self.TOGETHER_API_KEY,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )
        
        # Create conversation components
        self.setup_conversation()
        
    def setup_conversation(self):
        # Create message history store
        self.message_history = ChatMessageHistory()
        
        # Use ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            chat_memory=self.message_history,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
            human_prefix="Human",
            ai_prefix="AI",
        )
        
        # Define the prompt template
        self.combine_docs_prompt = PromptTemplate.from_template("""
        You are a highly knowledgeable AI assistant specializing in medical information retrieval.

        Previous conversation: {chat_history}

        Human question: {question}

        Context information from documents:
        {context}
        

        Provide only the final answer to the question without showing your internal reasoning.
        """)
        
        # Create the conversational retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_model,
            retriever=self.retriever,
            memory=self.memory,
            verbose=False,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.combine_docs_prompt},
        )
        
        
    
    def process_query(self, user_input):
        try:
            # Process query through the conversation chain
            response = self.invoke({"question": user_input})
            return {
                "answer": response['answer'],
                "success": True,
                "sources": [doc.page_content[:200] + "..." for doc in response.get('source_documents', [])]
            }
        except Exception as e:
            logger.error(f"Error in primary processing: {str(e)}")
            
            # Fallback approach
            try:
                # Get relevant documents directly
                docs = self.retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Direct query to the model
                from langchain_core.messages import HumanMessage, SystemMessage
                
                result = self.llm_model.invoke([
                    SystemMessage(content="""You are a medical assistant. Answer based on the context provided. give answer in html format for formating of bold,heading, subheading. dont use h1 or h2 tag.use bullet for point in list item dont use * use '-' or unicode:&#8226 """),
                    HumanMessage(content=f"Question: {user_input}")
                ])
                print(result.content)
                return {
                    "answer": result.content,
                    "success": True,
                    "fallback": True,
                    "sources": [doc.page_content[:200] + "..." for doc in docs]
                }
            except Exception as e2:
                logger.error(f"Fallback processing also failed: {str(e2)}")
                return {
                    "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
                    "success": False,
                    "error": str(e2)
                }
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.setup_conversation()
        return {"status": "Conversation history has been reset."}

