import os
from dotenv import load_dotenv

import operator
from typing import Annotated, List, Any, Dict
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage, AnyMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.checkpoint.sqlite import SqliteSaver

from typing_extensions import TypedDict

import sqlite3

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY') 
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage], operator.add]

conn = sqlite3.connect('checkpoints.db', check_same_thread=False)
memory = SqliteSaver(conn)

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        
        graph = StateGraph(AgentState)

        graph.add_node("llm", self.call_gemini)

        graph.add_node("action", self.take_action)

        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})

        graph.add_edge("action", "llm")

        graph.set_entry_point("llm")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_gemini(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        print("Mensagens enviadas ao modelo:", messages)
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling Tool: {t['name']} with args: {t['args']}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Returning to LLM after action!")
        return {'messages': results}
      
current_tavily_api_key = os.getenv('TAVILY_API_KEY')
if not current_tavily_api_key:
    raise ValueError("TAVILY_API_KEY não encontrada. Certifique-se de que está no seu .env e python-dotenv está instalado.")

tool = TavilySearch(max_results=3, tavily_api_key=current_tavily_api_key)


prompt_system = """Você é um assistente de pesquisa inteligente. Use o mecanismo de busca (tavily_search_results_json) para procurar informações.
Você tem permissão para fazer múltiplas chamadas à ferramenta (em conjunto ou em sequência).
Busque informações apenas quando tiver certeza do que procurar.
Se precisar de mais detalhes para formular uma pergunta de acompanhamento, você tem permissão para fazer isso.
Quando solicitado a comparar informações (ex: qual é mais quente, maior, etc.), use as informações do histórico da conversa e dos resultados das ferramentas."""

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

abot = Agent(model, [tool], system=prompt_system, checkpointer=memory)

messages = [HumanMessage(content="Como está o tempo em São Paulo hoje (09/04/2026)?")]
thread = {"configurable": {"thread_id": "1"}} 

print("\n--- Pergunta 1: Tempo em São Paulo ---")
for event in abot.graph.stream({"messages": messages}, thread):
    for k, v in event.items():
        if k in ("llm", "action"): 
             print(f"{k}: {v['messages']}")
             
messages = [HumanMessage(content="E no Rio de Janeiro?")]
thread = {"configurable": {"thread_id": "1"}} 

print("\n--- Pergunta 2: Tempo no Rio de Janeiro ---")
for event in abot.graph.stream({"messages": messages}, thread):
    for k, v in event.items():
        if k in ("llm", "action"):
            print(f"{k}: {v['messages']}")
            
messages = [HumanMessage(content="Qual está mais quente?")]
thread = {"configurable": {"thread_id": "1"}} 

print("\n--- Pergunta 3: Comparação ---")
for event in abot.graph.stream({"messages": messages}, thread):
    for k, v in event.items():
        if k in ("llm", "action"):
            print(f"{k}: {v['messages']}")
            
messages = [HumanMessage(content="Qual está mais quente?")]
thread = {"configurable": {"thread_id": "2"}} 

print("\n--- Pergunta 4: Comparação ---")
for event in abot.graph.stream({"messages": messages}, thread):
    for k, v in event.items():
        if k in ("llm", "action"):
            print(f"{k}: {v['messages']}")