import logging
import os
from pathlib import Path
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from supabase_module import SupabaseModule
from core.utils import set_collection_name
from core.embeddings import get_embeddings
import config
from agent_module.email_module import send_email_notification

logger = logging.getLogger("agent_system")

def get_llm(model_name: str | None = None):
    """
    Returns the initialized LangChain LLM object using the ModelRegistry.
    """
    from agent_module.providers import create_llm
    m_name = model_name or config.DEFAULT_CHAT_MODEL
    logger.info("Retrieving LLM from registry: %s", m_name)
    try:
        model_config = config.MODEL_REGISTRY.get(m_name)
        return create_llm(model_config)
    except Exception as e:
        logger.error("Failed to get model '%s' from registry: %s", m_name, e)
        raise e

class AutonomousRAGAgent:
    def __init__(self, project_name: str, model_name: str | None = None, job_id: str | None = None):
        self.project_name = project_name
        self.model_name = model_name or config.DEFAULT_CHAT_MODEL
        self.job_id = job_id
        self.collection_name = set_collection_name(project_name)
        self.embeddings = get_embeddings()
        
        # Initialize LLM using the registry
        self.llm = get_llm(self.model_name)
        logger.info("AutonomousRAGAgent initialized │ project=%s │ model=%s", 
                    project_name, self.model_name)
        
        # Initialize vectorstore connection (Supabase)
        self.supabase_module = SupabaseModule()
        self.vectorstore = self.supabase_module.get_vectorstore(self.collection_name, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5, "filter": {"collection": self.collection_name}})
        
        # Create Retriever Tool with Logging
        def search_documents(query: str) -> str:
            logger.info("Agent Retriever │ Query: '%s' │ Collection: '%s'", query, self.collection_name)
            docs = self.retriever.invoke(query)
            logger.info("Agent Retriever │ Fetched %d documents from collection '%s'", len(docs), self.collection_name)
            return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

        self.retriever_tool = Tool(
            name="project_document_retriever",
            description="Database of documents and resources that explain every detail about the tender project.",
            func=search_documents
        )
        self.tools = [self.retriever_tool]
        
        # Define Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful assistant in analyzing projects tenders.\n\n"
             "You have access to tools that provide access to documents and resources that contains all the details and information about the project. "
             "Your should search and extract accurate information directly from these documents to answer the user's questions.\n\n"
             "The users questions contains description of what the data is about,use it as a guidance to respond.\n\n"
             "Guidelines:\n"
             "- Respond every question ONLY with the information available in the database.\n"
             "- Respond with maximum 20 words per question.\n"
             "- respond only in Spanish.\n"
             "- Minimize reasoning steps. Use the retriever tool to search once, extract the exact answer, and provide the answer immediately without redundant or multiple searches."
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create Agent
        try:
            model_config = config.MODEL_REGISTRY.get(self.model_name)
            is_gemini = model_config.provider == "gemini"
            if is_gemini:
                logger.info("Gemini provider detected. Forcing ReAct agent due to tool binding capabilities.")
                raise NotImplementedError("Forcing ReAct for Gemini.")
            
            self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
            logger.info("Created tool-calling agent successfully.")
        except (NotImplementedError, AttributeError) as e:
            logger.info("Model does not support native tool calling (bind_tools) or fallback requested. Falling back to ReAct agent. Error: %s", e)
            self.prompt = ChatPromptTemplate.from_template(
                "Eres un asistente experto en analizar pliegos de condiciones de licitaciones públicas.\n"
                "Tienes acceso a las siguientes herramientas para obtener información detallada sobre el proyecto:\n\n"
                "{tools}\n\n"
                "Para responder, debes seguir estrictamente el siguiente formato:\n\n"
                "Question: la pregunta del usuario que debes responder\n"
                "Thought: piensa paso a paso sobre qué información necesitas buscar\n"
                "Action: la acción a realizar, debe ser una de [{tool_names}]\n"
                "Action Input: el término de búsqueda exacto\n"
                "Observation: el resultado de la herramienta\n"
                "... (este ciclo de Thought/Action/Action Input/Observation se puede repetir si es necesario)\n"
                "Thought: ya tengo la respuesta final clara a partir de los documentos\n"
                "Final Answer: la respuesta directa en español (máximo 20 palabras)\n\n"
                "Reglas importantes:\n"
                "- Responde únicamente con información verídica extraída de los documentos del proyecto.\n"
                "- La respuesta en 'Final Answer' debe ser concisa, directa y profesional en español.\n"
                "- Minimiza los pasos de razonamiento. Usa la herramienta de búsqueda una sola vez para encontrar la respuesta exacta y responde de inmediato en 'Final Answer' sin realizar búsquedas redundantes.\n\n"
                "Comencemos:\n\n"
                "Question: {input}\n"
                "Thought: {agent_scratchpad}"
            )
            self.agent = create_react_agent(self.llm, self.tools, self.prompt)
            logger.info("Created ReAct agent successfully.")

        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, max_iterations=15, handle_parsing_errors=True)

    def process_prompts(self, prompts: list[str]) -> str:
        import time
        
        categories = {
            "Generales": [0, 1, 2, 3, 5],
            "Requisitos Económicos": [6, 7, 18, 19, 20, 21],
            "Requisitos Técnicos": [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        }
        
        full_report_md = ""
        answers = {}
        email_body = None
        
        start_all = time.perf_counter()
        logger.info("Agent  │  Processing %d prompts sequentially (preventing rate limit and usage spikes)...", len(prompts))

        if self.job_id:
            try:
                self.supabase_module.update_pipeline_job(
                    self.job_id,
                    status="report_progress",
                    total_prompts=len(prompts),
                    completed_prompts=0
                )
            except Exception as e:
                logger.error("Failed to update initial report progress in Supabase: %s", e)

        for idx, prompt_text in enumerate(prompts):
            logger.info("Agent  │  Processing prompt %d/%d: '%s...'", idx + 1, len(prompts), prompt_text[:40])
            
            if self.job_id:
                try:
                    self.supabase_module.update_pipeline_job(
                        self.job_id,
                        status="report_progress",
                        current_prompt=prompt_text
                    )
                except Exception as e:
                    logger.error("Failed to update current_prompt in Supabase: %s", e)
            
            start_t = time.perf_counter()
            try:
                response = self.agent_executor.invoke({"input": prompt_text})
                answers[idx] = response["output"].strip()
                logger.info("Agent  │  Completed prompt %d/%d in %.2fs", idx + 1, len(prompts), time.perf_counter() - start_t)
            except Exception as e:
                logger.error("Agent  │  Critical failure on prompt %d: %s", idx + 1, e)
                answers[idx] = f"*Error processing this section: {str(e)}*"
            
            if self.job_id:
                try:
                    self.supabase_module.update_pipeline_job(
                        self.job_id,
                        status="report_progress",
                        completed_prompts=idx + 1
                    )
                except Exception as e:
                    logger.error("Failed to update completed_prompts in Supabase: %s", e)

        # --- Sequential Email Body Generation ---
        logger.info("Agent  │  Generating email body...")
        start_t = time.perf_counter()
        try:
            email_prompt = (
                f"Presenta este correo electrónico de manera profesional. "
                f"Explica que se trata de un resumen de una licitación. "
                f"Proporciona un resumen de exactamente 20 palabras sobre de qué trata el proyecto '{self.project_name}' "
                f"basándote en la información analizada."
            )
            email_body_response = self.llm.invoke(email_prompt)
            email_body = email_body_response.content.strip()
            logger.info("Agent  │  Completed email body generation in %.2fs", time.perf_counter() - start_t)
        except Exception as e:
            logger.error("Agent  │  Critical failure on email body: %s", e)
            email_body = f"Failed to generate email body: {str(e)}"

        logger.info("Agent  │  All sequential RAG tasks completed in %.2fs!", time.perf_counter() - start_all)

        # Assemble the final report based on defined categories
        for cat_name, indices in categories.items():
            full_report_md += f"## {cat_name}\n\n"
            for idx in indices:
                if idx in answers and answers[idx]:
                    full_report_md += f"{answers[idx]}\n\n"
        
        # --- Email Sending Logic ---
        if 0 in answers and email_body:
            try:
                subject = answers[0]
                logger.info("Agent  │  Sending email notification with subject: %s", subject[:50])
                send_email_notification(subject=subject, message=email_body)
            except Exception as e:
                logger.error("Agent  │  Failed to send email notification: %s", e)
                    
        return full_report_md
