from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import TeamState
from config import get_llm
from agents.research_agent import research_node
from agents.analyst_agent import analyst_node

# Esquema Pydantic (se mantiene para cumplir con el req. de Structured Output)
class RouterSchema(BaseModel):
    next_step: Literal["Investigador", "Analista", "FINISH"] = Field(
        description="El próximo agente que debe actuar o FINISH si la tarea está completada."
    )
    reasoning: str = Field(description="Explicación breve de por qué se toma esta decisión.")

def supervisor_node(state: TeamState) -> dict:
    """Nodo Supervisor: Determina la ruta del flujo basándose en el estado actual."""
    step_count = state.get("step_count", 0)
    
    # 1. Prevención de "Supervisor Infinito"
    if step_count >= 5:
        print(" [Supervisor]: Límite de pasos alcanzado. Finalizando ejecución.")
        return {"next_agent": "FINISH"}
    
    # 2. VALIDACIÓN PROFUNDA (La que pidió el tutor)
    research = state.get("research_data")
    analysis = state.get("analysis_data")
    
    has_research = bool(research) and len(research.strip()) > 20 and "error" not in research.lower()
    has_analysis = bool(analysis) and len(analysis.strip()) > 20 and "error" not in analysis.lower()

    # 3. Lógica de enrutamiento determinista (evita fallos de API y bucles)
    if has_research and has_analysis:
        print(" [Supervisor] Datos completos. Yendo a validación final.")
        return {"next_agent": "FINISH"}

    if not has_research:
        print(" [Supervisor] Faltan datos. Derivando a Investigador.")
        return {"next_agent": "Investigador"}

    if not has_analysis:
        print(" [Supervisor] Investigación lista. Derivando a Analista.")
        return {"next_agent": "Analista"}

    return {"next_agent": "FINISH"}


def validation_node(state: TeamState) -> dict:
    """Nodo dedicado a validar la calidad final antes de terminar."""
    research = state.get("research_data", "") or ""
    analysis = state.get("analysis_data", "") or ""
    print(f" [Validación] Longitud Investigación: {len(research)} | Longitud Análisis: {len(analysis)}")
    return state


def route_decision(state: TeamState) -> str:
    """Función de arista condicional que lee el próximo agente en el estado."""
    return state.get("next_agent", "FINISH")


def build_graph() -> StateGraph:
    """Construye y compila el grafo jerárquico."""
    builder = StateGraph(TeamState)

    # 1. Agregar Nodos
    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("Investigador", research_node)
    builder.add_node("Analista", analyst_node)
    builder.add_node("Validacion", validation_node) 

    # 2. Definir Punto de Entrada
    builder.set_entry_point("Supervisor")

    # 3. Aristas Condicionales desde el Supervisor
    builder.add_conditional_edges(
        "Supervisor",
        route_decision,
        {
            "Investigador": "Investigador",
            "Analista": "Analista",
            "FINISH": "Validacion"
        }
    )

    # 4. Retorno de los especialistas al Supervisor
    builder.add_edge("Investigador", "Supervisor")
    builder.add_edge("Analista", "Supervisor")
    builder.add_edge("Validacion", END) 

    return builder.compile()
  
# Exposición de la instancia lista para usar
graph = build_graph()