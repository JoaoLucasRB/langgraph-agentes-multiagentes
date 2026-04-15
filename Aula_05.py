import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Any, Dict
import operator
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dataclasses import dataclass, field
from uuid import uuid4

import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY') 
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

tool = TavilySearch(max_results=2)

def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    
    merged = left.copy()
    
    for message in right:
      for i, existing in enumerate(merged):
        if existing.id == message.id:
          merged[i] = message
          break
      else:
        merged.append(message)
          
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]
    
class Agent:

    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"] # Adiciona interrupção antes de chamar a ação
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_gemini(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        print(state)
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Chamando ferramenta: {t['name']} com argumentos: {t['args']}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Voltando para o modelo!")
        return {'messages': results}
      
from datetime import date
current_date = date.today().strftime("%d/%m/%Y") 

prompt = f"""Você é um assistente de pesquisa inteligente e altamente atualizado. \
Sua principal prioridade é encontrar as informações mais RECENTES e em TEMPO REAL sempre que possível. \
A data atual é {current_date}. \
Ao buscar sobre o tempo ou eventos que se referem a "hoje" ou "agora", \
você DEVE **incluir a data atual '{current_date}' na sua consulta para a ferramenta de busca**. \
Por exemplo, se a pergunta é "tempo em cidade x hoje", a consulta para a ferramenta deve ser "tempo em cidade x {current_date}". \
Ignore ou descarte informações que claramente se refiram a datas passadas ou futuras ao responder perguntas sobre "hoje". \
Use o mecanismo de busca para procurar informações, sempre buscando o 'hoje' ou o 'agora' quando o contexto indicar. \
Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza do que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!
"""

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

dynamic_thread_id = str(uuid4())

print(f"Meu novo Thead ID é: {dynamic_thread_id}")

from IPython.display import Image, display
import uuid

new_session_id = str(uuid4())
print(f"DEBUG: Iniciando nova conversa com ID: {new_session_id}\n")
new_user_message = "Qual é a distância entre o Rio de Janeiro e Tóquio?"
new_messages = [HumanMessage(content=new_user_message)]
new_thread_config = {"configurable": {"thread_id": new_session_id}}

print("--- Iniciando NOVA Interação: Agente processa a entrada e decide a ação ---")
print(f"Você: {new_user_message}")
print(f"DEBUG: Nova Thread ID: {new_session_id}")

print("\n--- Agente pensando e pausando ---")
try:
    for event in abot.graph.stream({"messages": new_messages}, new_thread_config):
        for k, v in event.items():
            if k == "llm":
                if v and 'messages' in v and v['messages']:
                    llm_message_from_event = v['messages'][0]
                    if hasattr(llm_message_from_event, 'tool_calls') and llm_message_from_event.tool_calls:
                        print(f"\nAgente (decisão): {llm_message_from_event.tool_calls}")
                        print("\n--- AGENTE PAUSADO: Intervenção Humana Necessária ---")
                        
                    elif llm_message_from_event.content:
                        print(f"\nAgente (resposta direta): {llm_message_from_event.content}")
                        print("\n--- AGENTE NÃO PAUSOU PARA FERRAMENTA (Resposta direta do LLM) ---")
except Exception as e:
    print(f"DEBUG: Stream interrompido como esperado: {e}")

current_state_snapshot = abot.graph.get_state(new_thread_config)

if current_state_snapshot:
    print(f"\nDEBUG: Estado atual obtido para NOVA thread ID: {new_session_id}")
    
    snapshot_thread_id = None
    snapshot_thread_ts = None

    if hasattr(current_state_snapshot, 'config') and isinstance(current_state_snapshot.config, dict):
        if 'configurable' in current_state_snapshot.config and isinstance(current_state_snapshot.config['configurable'], dict):
            if 'thread_id' in current_state_snapshot.config['configurable']:
                snapshot_thread_id = current_state_snapshot.config['configurable']['thread_id']
            if '__run_id' in current_state_snapshot.config['configurable']:
                snapshot_thread_ts = current_state_snapshot.config['configurable']['__run_id']
            elif 'thread_ts' in current_state_snapshot.config['configurable']:
                snapshot_thread_ts = current_state_snapshot.config['configurable']['thread_ts']
    
    if snapshot_thread_id is None:
        snapshot_thread_id = new_session_id

    print(f"DEBUG: ID da Thread (do snapshot): {snapshot_thread_id}")
    print(f"DEBUG: Timestamp do snapshot (thread_ts): {snapshot_thread_ts}") 
    print(f"DEBUG: Mensagens no snapshot (no momento da pausa): {current_state_snapshot.values.get('messages')}")
    
    if current_state_snapshot.values and 'messages' in current_state_snapshot.values:
        last_msg_in_snapshot = current_state_snapshot.values['messages'][-1]
        print(f"DEBUG: Tipo da última mensagem no snapshot para injeção: {type(last_msg_in_snapshot)}")
        if hasattr(last_msg_in_snapshot, 'tool_calls') and last_msg_in_snapshot.tool_calls:
            print(f"DEBUG: Última mensagem no snapshot TEM tool_calls. PRONTO PARA INJEÇÃO!")
        else:
            print(f"DEBUG: Última mensagem no snapshot NÃO TEM tool_calls ou está vazia. PROBLEMA NA PAUSA!")
    if current_state_snapshot.next != ():
        print("\n--- Agente está PAUSADO e pronto para intervenção. ---")
    else:
        print("\n--- ATENÇÃO: O agente NÃO está pausado onde esperávamos. O grafo pode ter terminado. ---")
else:
    print(f"DEBUG: Nenhum estado encontrado para a nova thread ID: {new_session_id}. Verifique a configuração da thread ou se o agente pausou.")
    
if current_state_snapshot:
    modified_state_values = current_state_snapshot.values.copy()

    final_injected_message = AIMessage(
        content="A distância entre o Rio de Janeiro e Tóquio é de aproximadamente 450 km. (Dados fornecidos MANULMENTE por você!)"
    )
    ai_message_found = False
    for i, msg in enumerate(modified_state_values['messages']):
        
        if isinstance(msg, AIMessage):
            modified_state_values['messages'] = modified_state_values['messages'][:i] + [final_injected_message]
            ai_message_found = True
            break
            
    if not ai_message_found:
        modified_state_values['messages'].append(final_injected_message)

    print("\n--- Estado sendo MODIFICADO MANUALMENTE (Injetando AIMessage Final) ---")
    print(f"DEBUG: Conteúdo da AIMessage falsa injetada: {final_injected_message.content}")
    print(f"DEBUG: Nova lista de mensagens (últimas): {[m.type for m in modified_state_values['messages'][-2:]]}")

else:
    print("DEBUG: Não é possível modificar o estado porque nenhum snapshot do estado foi encontrado.")
    
print("\n--- Finalizando o estado com a resposta injetada ---")

abot.graph.update_state(new_thread_config, modified_state_values)

final_state_after_injection_obj = abot.graph.get_state(new_thread_config)

print("\n--- Saída final do agente após intervenção ---")

if hasattr(final_state_after_injection_obj, 'values') and isinstance(final_state_after_injection_obj.values, dict):
    final_messages = final_state_after_injection_obj.values['messages']
elif isinstance(final_state_after_injection_obj, dict):
    final_messages = final_state_after_injection_obj['messages']

else:
    found_messages_list = None
    if isinstance(final_state_after_injection_obj, tuple):
        for item in final_state_after_injection_obj:
            if isinstance(item, dict) and 'messages' in item:
                found_messages_list = item['messages']
                break
    elif isinstance(final_state_after_injection_obj, dict) and 'messages' in final_state_after_injection_obj:
        found_messages_list = final_state_after_injection_obj['messages']
    
    if found_messages_list is not None:
        final_messages = found_messages_list
    else:
        print(f"DEBUG: Não foi possível extrair a lista de mensagens do objeto de estado final: {final_state_after_injection_obj}")
        final_messages = []

if final_messages and isinstance(final_messages[-1], AIMessage):
    print(f"\nAgente: {final_messages[-1].content}")
else:
    print("\nAgente: Resposta final não encontrada ou não é um AIMessage.")
    print(f"DEBUG: Estado final completo (para inspeção): {final_state_after_injection_obj}")

print("\n--- Fluxo de Human-in-the-Loop concluído ---")