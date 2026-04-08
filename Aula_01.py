import os
from dotenv import load_dotenv
import re
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
client = genai.GenerativeModel("gemini-2.5-flash")
PROMPT_REACT = """
Você funciona em um ciclo de Pensamento, Ação, Pausa e Observação.
Ao final do ciclo, você fornece uma Resposta.
Use "Pensamento" para descrever seu raciocínio.
Use "Ação" para executar ferramentas - e então retorne "PAUSA".
A "Observação" será o resultado da ação executada.
Ações disponíveis:
  - consultar_estoque: retorna a quantidade disponível de um item no inventário (ex: "consultar_estoque: teclado")
  - consultar_preco_produto: retorna o preço unitário de um produto (ex: "consultar_preco_produto: mouse gamer")
  - encontrar_produto_mais_caro: retorna o nome e preço do produto mais caro do inventário (não requer argumentos, ex: "encontrar_produto_mais_caro")
  - calcular_valor_total_lista: calcula o valor total de uma lista de itens (ex: "calcular_valor_total_lista: teclado, mouse gamer, monitor")

Exemplo:
Pergunta: Quantos monitores temos em estoque?
Pensamento: Devo consultar a ação consultar_estoque para saber a quantidade de monitores.
Ação: consultar_estoque: monitor
PAUSA

Observação: Temos 75 monitores em estoque.
Resposta: Há 75 monitores em estoque.

Exemplo:
Pergunta: Qual é o produto mais caro?
Pensamento: Preciso usar a ação encontrar_produto_mais_caro para descobrir qual produto tem o maior preço.
Ação: encontrar_produto_mais_caro
PAUSA

Observação: O produto mais caro é o(a) monitor com preço de R$ 999.90.
Resposta: O produto mais caro é o(a) monitor com preço de R$ 999.90.

Exemplo:
Pergunta: Quanto custa um teclado e um mouse gamer?
Pensamento: O usuário quer saber o valor total de vários itens. Devo usar a ação calcular_valor_total_lista com os itens "teclado, mouse gamer".
Ação: calcular_valor_total_lista: teclado, mouse gamer
PAUSA

Observação: O valor total dos itens encontrados é R$ 249.50.
Resposta: O valor total do teclado e do mouse gamer é R$ 249.50.
""".strip()

class Agent:
  def __init__(self, system=""):
    self.system = system
    self.messages = []
    if self.system:
      self.messages.append({"role": "system", "content": self.system})
      
  def __call__(self, message):
    self.messages.append({"role": "user", "content": message})
    result = self.execute()
    self.messages.append({"role": "assistant", "content": result})
    return result
  
  def execute(self):
    prompt = ""
    for msg in self.messages:
      prompt += f"{msg['role']}: {msg['content']}\n"
      
    response = client.generate_content(prompt)
    return response.text
  
class AgentState(TypedDict):
  pergunta: str
  historico: list[str]
  acao_pendente: str
  resposta_final: str
  
def consultar_estoque(item: str) -> str:
  item = item.lower()
  estoque = {
    "teclado": 150,
    "mouse gamer": 80,
    "monitor": 75,
    "impressora": 40,
    "webcam": 60,
    "headset": 30
  }
  
  if item in estoque:
    return f"Temos {estoque[item]} {item}s em estoque."
  else:
    return f"Item '{item}' não encontrado no inventário."
  
def consultar_preco_produto(item: str) -> str:
  item = item.lower()
  precos = {
    "teclado": 120.00,
    "mouse gamer": 85.50,
    "monitor": 450.00,
    "impressora": 300.00,
    "webcam": 150.00,
    "headset": 200.00
  }
  
  if item in precos:
    return f"O preço unitário do {item} é R${precos[item]:.2f}."
  else:
    return f"Produto '{item}' não encontrado no inventário."
  
def encontrar_produto_mais_caro() -> str:
  precos_do_inventario = {
    "teclado": 120.00,
    "mouse gamer": 85.50,
    "monitor": 450.00,
    "impressora": 300.00,
    "webcam": 150.00,
    "headset": 200.00
  }
  
  if not precos_do_inventario:
    return "Nenhum produto encontrado na lista de preços para comparação."
  
  nome_produto_mais_caro = max(precos_do_inventario, key=precos_do_inventario.get)
  valor_produto_mais_caro = precos_do_inventario[nome_produto_mais_caro]
  
  print(f"Debug: Produto mais caro encontrado - {nome_produto_mais_caro} com preço R${valor_produto_mais_caro:.2f}")
  
  return f"O produto mais caro é '{nome_produto_mais_caro}' com preço de R${valor_produto_mais_caro:.2f}."

def calcular_valor_total_lista(lista_itens: str) -> str:
  precos_do_inventario = {
    "teclado": 120.00,
    "mouse gamer": 85.50,
    "monitor": 450.00,
    "impressora": 300.00,
    "webcam": 150.00,
    "headset": 200.00
  }
  
  itens_processados = [item.strip().lower() for item in lista_itens.split(",")]
  
  valor_total = 0.0
  itens_nao_encontrados = []
  
  for item in itens_processados:
    if item in precos_do_inventario:
      valor_total += precos_do_inventario[item]
    else:
      itens_nao_encontrados.append(item)
      
  resposta = f"O valor total da lista de itens é R${valor_total:.2f}."
  
  if itens_nao_encontrados:
    resposta += f" Os seguintes itens não foram encontrados no inventário: {', '.join(itens_nao_encontrados)}."
  
  return resposta

def run_react_agent(pergunta: str, max_iterations: int = 5) -> str:
  model = genai.GenerativeModel('gemini-2.5-flash')

  chat = model.start_chat(history=[])
  chat.send_message(PROMPT_REACT)

  current_prompt = pergunta

  for i in range(max_iterations):
    response = chat.send_message(current_prompt)
    response_text = response.text.strip()

    print(f"\n--- Iteração {i+1} ---")
    print(f"Modelo pensou/respondeu:\n{response_text}\n")

        
    response_match_final = re.search(r"Resposta:\s*(.*)", response_text, re.DOTALL)
    if response_match_final:
      return response_match_final.group(1).strip()

        
    match = re.search(r"Ação:\s*(\w+)(?::\s*([^\n]*))?", response_text)

    if match:
      action_name = match.group(1).strip()

      action_arg = match.group(2).strip() if match.group(2) is not None else ""

      observacao_da_acao = ""

      if action_name == "consultar_estoque":
        observacao_da_acao = consultar_estoque(action_arg)
      elif action_name == "consultar_preco_produto":
        observacao_da_acao = consultar_preco_produto(action_arg)
      elif action_name == "encontrar_produto_mais_caro": 
        observacao_da_acao = encontrar_produto_mais_caro()
      elif action_name == "calcular_valor_total_lista":
        observacao_da_acao = calcular_valor_total_lista(action_arg)
      else:
        observacao_da_acao = f"Erro: Ação '{action_name}' desconhecida. Verifique o prompt ou a implementação da ferramenta."

      current_prompt = f"Observação: {observacao_da_acao}"

      print(f"Executou ação: {action_name} com argumento '{action_arg}'")
      print(f"Observação: {observacao_da_acao}\n")

    else:          
      return f"Erro: O agente não conseguiu extrair uma Ação ou Resposta final após {i+1} iterações. Última resposta do modelo: {response_text}"
  return "Erro: Limite máximo de iterações atingido sem uma resposta final do agente."

def iniciar_conversacao_com_agente():
  print("--- Agente de Inventário Interativo ---") 
  print("Digite sua pergunta sobre o inventário ou digite 'sair' para encerrar a conversa.")
  print("-"*50)
  
  while True:
    pergunta_usuario = input("Você: ").strip()
    
    if pergunta_usuario.lower() == "sair":
      print("Encerrando a conversa. Até mais!")
      break
    
    print("\nProcessando sua pergunta...  Aguarde um momento.\n")
    
    try:
      resposta_agente = run_react_agent(pergunta_usuario)
      print(f"Agente: {resposta_agente}\n")
    except Exception as e:
      print(f"Erro ao processar a pergunta: {e}\n")
      print("Tente novamente ou digite 'sair' para encerrar a conversa.\n")
  
if __name__ == "__main__":
  iniciar_conversacao_com_agente()