import logging
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from supabase_module import SupabaseModule
from core.utils import set_collection_name
from core.embeddings import get_embeddings
from agent_module.agent_system import get_llm
import config

logger = logging.getLogger("rag_responder")

class RAGResponder:
    """
    A versatile RAG responder module that works independently of report generation.
    """
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.DEFAULT_CHAT_MODEL
        self.llm = get_llm(self.model_name)
        self.embeddings = get_embeddings()

    def get_project_title(self, project_name: str, chunks: List[Document] = None) -> str:
        """
        Identify the natural language project title of the tender.
        If chunks are provided, uses the first few chunks to extract the title.
        Otherwise, queries the vector store to retrieve relevant text and extract the title.
        """
        text_content = ""
        if chunks:
            logger.info("RAGResponder  │  Naming project '%s' using in-memory chunks", project_name)
            selected_chunks = chunks[:3]
            text_content = "\n\n".join([c.page_content for c in selected_chunks])
        else:
            logger.info("RAGResponder  │  Naming project '%s' by querying Supabase", project_name)
            collection_name = set_collection_name(project_name)
            try:
                supabase_module = SupabaseModule()
                vectorstore = supabase_module.get_vectorstore(collection_name, self.embeddings)
                search_query = "objeto de la licitación nombre del proyecto título"
                query_embedding = self.embeddings.embed_query(search_query)

                res = supabase_module.client.rpc(
                    vectorstore.query_name,
                    {
                        "query_embedding": query_embedding,
                        "match_count": 3,
                        "filter": {"collection": collection_name}
                    }
                ).execute()

                text_content = "\n\n".join([item.get("content", "") for item in res.data])
            except Exception as e:
                logger.error("RAGResponder  │  Failed to query Supabase for project naming: %s", e)
        
        if not text_content:
            logger.warning("RAGResponder  │  No content found to name project. Returning folder name.")
            return project_name

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Eres un asistente experto en licitaciones públicas.\n"
             "Tu tarea es identificar el nombre oficial o título del proyecto de la licitación a partir del fragmento de texto proveído.\n"
             "Reglas:\n"
             "- Responde ÚNICAMENTE con el nombre/título del proyecto, sin preámbulos, explicaciones ni formato adicional.\n"
             "- El título debe ser conciso pero descriptivo en español.\n"
             "- Si no encuentras un nombre explícito, resume el objeto principal en menos de 10 palabras (ej. 'Construcción de Acueducto')."
            ),
            ("human", "Texto del pliego:\n{text}\n\nIdentifica el nombre del proyecto:")
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({"text": text_content[:6000]})
            title = response.content.strip()
            title = title.replace('"', '').replace("'", "").replace("**", "").strip()
            for prefix in ["nombre del proyecto:", "título del proyecto:", "proyecto:", "objeto:"]:
                if title.lower().startswith(prefix):
                    title = title[len(prefix):].strip()
            logger.info("RAGResponder  │  Identified project title: '%s'", title)
            return title
        except Exception as e:
            logger.error("RAGResponder  │  Failed to generate project title via LLM: %s", e)
            return project_name

    def respond_chat(self, project_name: str, question: str) -> dict:
        """
        Perform a RAG search on the project's vector store and answer the question.
        """
        logger.info("RAGResponder  │  Responding to chat question for project '%s': '%s'", project_name, question)
        collection_name = set_collection_name(project_name)
        try:
            supabase_module = SupabaseModule()
            vectorstore = supabase_module.get_vectorstore(collection_name, self.embeddings)
            query_embedding = self.embeddings.embed_query(question)

            res = supabase_module.client.rpc(
                vectorstore.query_name,
                {
                    "query_embedding": query_embedding,
                    "match_count": 5,
                    "filter": {"collection": collection_name}
                }
            ).execute()

            sources = []
            context = ""
            for i, item in enumerate(res.data):
                doc_idx = i + 1
                metadata = item.get('metadata', {})
                content = item.get('content', '')
                
                source_file = metadata.get('source_file', 'Desconocido')
                page = metadata.get('page')
                
                # Get public URL if possible (assuming bucket name is 'project_files')
                object_name = f"{project_name}/{source_file}"
                file_url = supabase_module.get_public_url("project_files", object_name)
                
                sources.append({
                    "id": doc_idx,
                    "sourceFile": source_file,
                    "page": page,
                    "content": content[:300] + "..." if len(content) > 300 else content,
                    "fileUrl": file_url
                })
                context += f"\n\nDocumento [{doc_idx}]:\n{content}"
        except Exception as e:
            logger.exception("RAGResponder  │  Failed to query Supabase for chat message")
            return {"response": "Lo siento, hubo un error al consultar la base de datos del proyecto.", "sources": []}

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Eres un asistente inteligente especializado en responder preguntas sobre pliegos de condiciones de licitaciones.\n"
             "Responde la pregunta del usuario utilizando ÚNICAMENTE la información provista en el contexto.\n"
             "Si la respuesta no se encuentra en el contexto, indícalo de manera de forma educada.\n"
             "Reglas:\n"
             "- Responde de manera clara, concisa y profesional.\n"
             "- Responde en español.\n"
             "- Mantén la respuesta enfocada en los datos reales del pliego.\n"
             "- Cita SIEMPRE la información extraída de los documentos usando el formato de corchetes con el número del documento, por ejemplo [1], [2]."
            ),
            ("human", "Contexto del proyecto:\n{context}\n\nPregunta: {question}")
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({"context": context, "question": question})
            return {"response": response.content.strip(), "sources": sources}
        except Exception as e:
            logger.error("RAGResponder  │  Failed to generate chat response: %s", e)
            return {"response": "Lo siento, ocurrió un error interno al generar la respuesta.", "sources": sources}
