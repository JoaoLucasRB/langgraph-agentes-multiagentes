# app.py
import gradio as gr
from Aula_06 import graph  # Importa o grafo do seu novo backend
import uuid

# --- Função que será chamada pelo Gradio para rodar o agente ---
def generate_essay(topic: str, max_revisions: int):
    """
    Roda o grafo do agente para gerar uma redação e transmite as saídas em tempo real.
    """
    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        'task': topic,
        "max_revisions": max_revisions,
        "revision_number": 0,
        "plan": "",
        "draft": "",
        "critique": "",
        "content": []
    }

    full_output = ""
    # Itera sobre o stream do grafo para obter as saídas passo a passo
    for s in graph.stream(initial_state, thread_config):
        # A API do LangGraph retorna um dicionário de dicionários
        step_output = list(s.values())[0]

        # Formata a saída para ser mais legível na interface
        if 'plan' in step_output:
            full_output += f"### 📝 Plano Gerado:\n{step_output['plan']}\n\n"
        elif 'content' in step_output:
            # Exibe o conteúdo da pesquisa
            search_content = "\n".join(step_output['content'])
            full_output += f"### 🔍 Conteúdo de Pesquisa:\n{search_content}\n\n"
        elif 'draft' in step_output:
            full_output += f"### ✍️ Rascunho Gerado:\n{step_output['draft']}\n\n"
        elif 'critique' in step_output:
            full_output += f"### 🧐 Crítica e Revisão:\n{step_output['critique']}\n\n"

        # Adiciona uma linha divisória para separar os passos
        full_output += "---" * 20 + "\n\n"

        yield full_output
    
    yield full_output

# --- Criação da Interface Gradio ---
with gr.Blocks(theme=gr.themes.Default(spacing_size='sm', text_size="sm")) as demo:
    gr.Markdown("# 🤖 Gerador de Redações com Gemini e LangGraph")
    gr.Markdown(
        "Digite o tópico da sua redação e o número de revisões. "
        "O agente vai planejar, pesquisar, rascunhar e revisar o texto."
    )

    with gr.Row():
        essay_topic = gr.Textbox(label="Tópico da Redação", placeholder="Ex: A importância da inteligência artificial na educação")
        max_revisions_slider = gr.Slider(minimum=0, maximum=3, step=1, value=1, label="Número Máximo de Revisões")
        generate_button = gr.Button("Gerar Redação", variant="primary")

    output_textbox = gr.Textbox(label="Processo e Redação Final", lines=20, max_lines=40)

    # Associa o botão à função Python
    generate_button.click(
        fn=generate_essay,
        inputs=[essay_topic, max_revisions_slider],
        outputs=output_textbox
    )

# Lança a interface
if __name__ == "__main__":
    demo.launch(share=False)