import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel
from typing import Literal

load_dotenv()

class RouterSchema(BaseModel):
    next_step: Literal["Investigador", "Analista", "FINISH"]
    reasoning: str

class MockLLM(BaseChatModel):
    """MockLLM perfecto para probar el flujo del orquestador sin API."""
    
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        last_message = messages[-1]
        
        # 1. Respuesta a una herramienta
        if last_message.type == "tool":
            text = "Investigación y análisis completados con datos del Mock."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        # 2. Simulamos la decisión del Supervisor (Lee el system prompt)
        if isinstance(last_message, SystemMessage) and "Supervisor" in last_message.content:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        # 3. Agente decidiendo usar herramienta
        content = last_message.content.lower() if last_message.content else ""
        
        if "investiga" in content:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="", additional_kwargs={"tool_calls": [{"name": "buscar_informacion_mercado", "args": {"tema": "langgraph"}, "id": "call_1"}]})
            )])
        elif "analiza" in content or "métricas" in content:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="", additional_kwargs={"tool_calls": [{"name": "calculate_metrics", "args": {"numbers": [10, 20, 30]}, "id": "call_2"}]})
            )])
            
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Procesado por el Mock."))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return self
        
    def invoke(self, input, config=None, **kwargs):
        # Interceptamos solo cuando el input es un string (para Structured Output del supervisor)
        if isinstance(input, str) and "estado actual" in input.lower():
            if "investigación: pendiente" in input.lower():
                return RouterSchema(next_step="Investigador", reasoning="Falta investigación.")
            elif "investigación: completado" in input.lower() and "análisis: pendiente" in input.lower():
                return RouterSchema(next_step="Analista", reasoning="Falta análisis.")
            else:
                return RouterSchema(next_step="FINISH", reasoning="Todo completado.")
        
        # Si es un diccionario con mensajes (Agentes), va al _generate normal
        return super().invoke(input, config, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock"

def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if provider == "gemini" and google_key and google_key != "tu_gemini_api_key":
        try: return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=temperature, google_api_key=google_key, max_retries=1)
        except: pass
            
    if provider == "groq" and groq_key and groq_key != "tu_groq_api_key":
        try: return ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature, groq_api_key=groq_key, max_retries=1)
        except: pass

    if provider == "openai" and openai_key and openai_key != "tu_openai_api_key":
        try: return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=openai_key, max_retries=1)
        except: pass

    print(" Ejecutando en modo 'Mock' (Simulación sin consumo de API).")
    return MockLLM()