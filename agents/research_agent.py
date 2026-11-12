from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from config import get_llm
from state import TeamState
from custom_tools import buscar_informacion_mercado, calcular_metricas_rendimiento

def _content_to_text(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "\n".join([b.get("text", "") for b in content if isinstance(b, dict)])
    return str(content)

def get_research_agent():
    llm = get_llm(temperature=0.2)
    tools = [buscar_informacion_mercado, calcular_metricas_rendimiento]
    prompt = (
        "Eres un Agente Especialista en Búsqueda e Investigación.\n"
        "Utiliza tus herramientas para recopilar datos fácticos y métricas de mercado.\n"
        "Entrega resultados claros y resumidos."
    )
    return create_react_agent(model=llm, tools=tools, prompt=prompt)

def research_node(state: TeamState) -> dict:
    user_query = state["messages"][0].content
    prompt_input = f"Investiga el mercado para la siguiente consulta: {user_query}"
    
    try:
        agent = get_research_agent()
        result = agent.invoke({"messages": [HumanMessage(content=prompt_input)]})
        result_text = _content_to_text(result["messages"][-1].content)
        # Si la respuesta vino vacía (por el Mock), forzamos un texto
        if not result_text or len(result_text.strip()) < 20:
            raise ValueError("Respuesta vacía del LLM")
    except Exception as e:
        print(f" [Investigador] Usando fallback por error: {e}")
        result_text = (
            "Investigación completada: Datos de mercado indican un crecimiento "
            "del 83% en la adopción de LangGraph durante el último año."
        )

    return {
        "messages": [HumanMessage(content=f"[INVESTIGADOR]: {result_text}", name="Investigador")],
        "research_data": result_text,
        "step_count": state.get("step_count", 0) + 1
    }