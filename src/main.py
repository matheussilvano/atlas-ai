from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# 1. Importe o CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

# importa sua função de workflow
from assistente_rh_workflow import run_workflow, WorkflowInput

app = FastAPI(title="Assistente RH API", version="1.0")

# 2. Adicione a configuração de CORS
# Isso permite que qualquer origem (como seu arquivo HTML) acesse a API.
# Para produção, você pode restringir as origens.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)


class Pergunta(BaseModel):
    question: str

@app.post("/ask")
async def ask_rh(pergunta: Pergunta):
    try:
        workflow_input = WorkflowInput(input_as_text=pergunta.question)
        result = await run_workflow(workflow_input)
        return {"answer": result["output_text"]}
    except Exception as e:
        # É uma boa prática logar o erro no servidor para depuração
        print(f"Ocorreu um erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))