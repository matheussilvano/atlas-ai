# Importa a função para carregar variáveis de ambiente do arquivo .env
from dotenv import load_dotenv
# Executa a função para carregar as variáveis (deve ser chamado antes de qualquer código que as utilize)
load_dotenv()

import os
import streamlit as st
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PIL import Image

# Opcional: Para depuração da versão do Streamlit
# st.write(f"Versão do Streamlit em execução: {st.__version__}")

# --- CONFIGURAÇÃO DE SEGURANÇA E INICIALIZAÇÃO ---

def configurar_api_key():
    """Configura a chave da API do Google a partir do st.secrets ou variáveis de ambiente."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
    except (KeyError, FileNotFoundError):
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("Chave da API do Google não encontrada. Configure-a em um arquivo .env ou nos segredos do Streamlit.")
            st.stop()

# --- CARREGAMENTO E PROCESSAMENTO DE DADOS (CACHE) ---

@st.cache_resource
def carregar_vector_store():
    """
    Carrega o JSON, processa os documentos e cria o banco de dados vetorial.
    """
    try:
        with open('employees.json', 'r', encoding='utf-8') as f:
            dados_json = json.load(f)
    except FileNotFoundError:
        st.error("Arquivo 'employees.json' não encontrado. Certifique-se de que ele está na mesma pasta do script.")
        return None

    documentos_processados = []
    for colaborador in dados_json:
        lista_experiencias = [
            f"- {exp.get('titulo', 'N/A')}: {exp.get('descricao', 'N/A')}"
            for exp in colaborador.get('experiencias_relevantes', [])
        ]
        texto_experiencias = "\n".join(lista_experiencias)
        
        texto_colaborador = f"""
        Nome: {colaborador.get('nome_completo', 'N/A')} (ID: {colaborador.get('id_colaborador', 'N/A')})
        Habilidades Técnicas: {', '.join(colaborador.get('habilidades', {}).get('hard_skills', []))}
        Habilidades Interpessoais: {', '.join(colaborador.get('habilidades', {}).get('soft_skills', []))}
        Idiomas: {', '.join([f"{idioma.get('lingua', 'N/A')} ({idioma.get('nivel', 'N/A')})" for idioma in colaborador.get('idiomas', [])])}
        Cargos Anteriores: {', '.join(colaborador.get('historico_cargos', []))}
        Experiências Notáveis:\n{texto_experiencias}
        """
        documentos_processados.append(Document(page_content=texto_colaborador, metadata={"source_id": colaborador.get('id_colaborador')}))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(documentos_processados, embeddings)
    return vector_store

# --- LÓGICA DA CADEIA DE CONVERSA (LANGCHAIN) ---

def obter_rag_chain(_retriever):
    """
    Cria e retorna a cadeia de RAG que leva em conta o histórico da conversa.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Considerando o histórico da conversa, gere uma pergunta de busca que possa ser entendida sem a necessidade do histórico. Não responda à pergunta, apenas a reformule se necessário, caso contrário, retorne-a como está."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, _retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente de RH chamado Atlas.AI. Sua função é encontrar colaboradores com base em suas habilidades, experiências e histórico. Responda à pergunta do usuário de forma natural e conversacional, usando os documentos fornecidos como contexto. Seja amigável, mas profissional e direto. Se a informação não estiver nos documentos, informe que não encontrou um perfil com as características solicitadas.\n\nContexto:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- INTERFACE GRÁFICA (STREAMLIT) ---

def inicializar_estado_sessao():
    """Inicializa as variáveis necessárias no st.session_state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        with st.status("Carregando a base de conhecimento... Aguarde!", expanded=True) as status:
            st.write("Processando perfis de colaboradores...")
            st.session_state.vector_store = carregar_vector_store()
            st.write("Criando embeddings e indexando...")
            status.update(label="Base de conhecimento carregada!", state="complete", expanded=False)
    if "rag_chain" not in st.session_state and st.session_state.vector_store:
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        st.session_state.rag_chain = obter_rag_chain(retriever)

def renderizar_interface():
    """Desenha a interface principal do chat na tela."""
    try:
        page_icon = Image.open("favicon.png")
    except FileNotFoundError:
        page_icon = "🤖" 
    
    st.set_page_config(page_title="Atlas.AI", page_icon=page_icon, layout="wide")
    
    # NOVO: Injeção de CSS para forçar o alinhamento da mensagem do usuário e personalizar seu estilo
    st.markdown("""
        <style>
            /* Alinha as mensagens do usuário à direita */
            .stChatMessage[data-testid="stChatMessageUser"] {
                display: flex;
                flex-direction: row-reverse;
                text-align: right;
            }
            /* Opcional: remove o avatar padrão se você estiver usando um avatar de imagem
               e não quiser que o "👤" apareça como fallback em algumas versões */
            .stChatMessage[data-testid="stChatMessageUser"] .st-dg { /* Seleciona o elemento que contém o avatar */
                display: none; 
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        logo = Image.open("logo.png")
        st.image(logo, width=200)
    except FileNotFoundError:
        st.title("Atlas.AI")
    st.markdown("##### Olá! Sou seu assistente para busca de talentos internos.")

    # Define os avatares para assistente e usuário
    assistant_avatar = "favicon.png" if os.path.exists("favicon.png") else "🤖"
    user_avatar = "user.png" if os.path.exists("user.png") else "👤" # Personalizado para user.png

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "assistant", "content": "Como posso te ajudar a encontrar um colaborador hoje? Você pode perguntar sobre habilidades, experiências ou cargos."})

    for message in st.session_state.chat_history:
        # Usa o avatar customizado para o assistente e para o usuário
        avatar_to_use = assistant_avatar if message["role"] == "assistant" else user_avatar
        with st.chat_message(name=message["role"], avatar=avatar_to_use): # O 'name' já define o alinhamento
            st.markdown(message["content"])


def processar_interacao_usuario():
    """Captura a entrada do usuário pelo chat_input e a adiciona ao histórico."""
    prompt = st.chat_input("Busque por habilidades, projetos, cargos...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

def mostrar_sugestoes():
    """Mostra botões com perguntas de exemplo se a conversa ainda não começou."""
    if len(st.session_state.chat_history) == 1:
        st.markdown("---")
        st.markdown("**Ou tente um destes exemplos:**")
        sugestoes = [
            "Quem tem experiência com Python e liderança de projetos?",
            "Encontre alguém que fale alemão fluente",
            "Qual colaborador trabalhou com otimização de campanhas de marketing?"
        ]
        cols = st.columns([1, 1, 1.2]) 
        for i, sugestao in enumerate(sugestoes):
            if cols[i].button(sugestao, use_container_width=True, key=f"sug_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": sugestao})
                st.rerun()

def gerar_resposta_ia():
    """Verifica se a última mensagem é do usuário e gera uma resposta da IA com streaming."""
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        assistant_avatar = "favicon.png" if os.path.exists("favicon.png") else "🤖"
        
        with st.chat_message("assistant", avatar=assistant_avatar):
            message_placeholder = st.empty()
            full_response = ""
            context_docs = []

            stream = st.session_state.rag_chain.stream({
                "input": st.session_state.chat_history[-1]["content"],
                "chat_history": st.session_state.chat_history
            })

            for chunk in stream:
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    message_placeholder.markdown(full_response + "▌")
                if "context" in chunk:
                    context_docs = chunk["context"]
            
            message_placeholder.markdown(full_response)
            
            if context_docs:
                with st.expander("Ver fontes consultadas", expanded=False):
                    for doc in context_docs:
                        st.code(doc.page_content, language="text")

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

def main():
    """Função principal que executa a aplicação."""
    configurar_api_key()
    inicializar_estado_sessao()
    
    if st.session_state.vector_store is None:
        st.warning("A base de dados de colaboradores não pôde ser carregada. Verifique o arquivo 'employees.json'.")
        st.stop()
        
    renderizar_interface()
    mostrar_sugestoes()
    processar_interacao_usuario()
    gerar_resposta_ia()

if __name__ == "__main__":
    main()