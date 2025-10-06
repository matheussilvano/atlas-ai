import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- 1. Configuração Inicial ---
# Lembre-se de colocar sua chave de API do Google AI Studio aqui.
os.environ["GOOGLE_API_KEY"] = "AIzaSyCZHCXfKzPS2CIZHO0N4fElCsZxNp1UlZQ"

# --- 2. Carregamento e Processamento do JSON ---
with open('employees.json', 'r', encoding='utf-8') as f:
    dados_json = json.load(f)

# Vamos criar um "documento" de texto para cada colaborador
documentos_processados = []
for colaborador in dados_json:
    # Formata uma string de texto coesa com as principais informações
    texto_colaborador = f"""
Nome: {colaborador['nome_completo']} (ID: {colaborador['id_colaborador']})
Habilidades Técnicas: {', '.join(colaborador['habilidades']['hard_skills'])}
Habilidades Interpessoais: {', '.join(colaborador['habilidades']['soft_skills'])}
Idiomas: {', '.join([f"{idioma['lingua']} ({idioma['nivel']})" for idioma in colaborador['idiomas']])}
Cargos Anteriores: {', '.join(colaborador['historico_cargos'])}
Experiências Notáveis: {', '.join(colaborador['experiencias_relevantes'])}
"""
    # Adiciona o texto formatado à lista como um objeto Documento do LangChain
    documentos_processados.append(Document(page_content=texto_colaborador))

print(f"{len(documentos_processados)} perfis de colaboradores carregados e processados.")

# --- 3. Criação dos Embeddings e do Banco Vetorial ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(documentos_processados, embeddings)
print("Banco de dados vetorial de colaboradores criado com sucesso!")

# --- 4. Configuração da Cadeia de Pergunta e Resposta (RAG) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
prompt_template = """
Você é um assistente de RH. Responda à pergunta do gestor de forma clara e objetiva,
usando estritamente as informações do contexto fornecido sobre os colaboradores.
Liste os nomes das pessoas que atendem ao critério.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# --- 5. Execução da Pergunta ---
def encontrar_colaborador(pergunta):
    documentos_similares = vector_store.similarity_search(pergunta, k=3) # Busca os 3 perfis mais relevantes
    resposta = chain(
        {"input_documents": documentos_similares, "question": pergunta},
        return_only_outputs=True
    )
    return resposta['output_text']

# --- Testando com perguntas do gestor ---
pergunta_gestor_1 = "Quem da minha equipe fala espanhol?"
pergunta_gestor_2 = "Preciso de alguém que já treinou clientes e tenha experiência com vendas. Quem pode me ajudar?"
pergunta_gestor_3 = "Quais colaboradores sabem usar a metodologia Scrum?"

print("\n--- PERGUNTA 1 ---")
print(f"Pergunta: {pergunta_gestor_1}")
print(f"Resposta: {encontrar_colaborador(pergunta_gestor_1)}")

print("\n--- PERGUNTA 2 ---")
print(f"Pergunta: {pergunta_gestor_2}")
print(f"Resposta: {encontrar_colaborador(pergunta_gestor_2)}")

print("\n--- PERGUNTA 3 ---")
print(f"Pergunta: {pergunta_gestor_3}")
print(f"Resposta: {encontrar_colaborador(pergunta_gestor_3)}")