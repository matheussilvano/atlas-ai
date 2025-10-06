from agents import FileSearchTool, RunContextWrapper, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

# Tool definitions
file_search = FileSearchTool(
  vector_store_ids=[
    "vs_68e4287fa6408191a8fe4521771b432c"
  ]
)
class AssistenteRhContext:
  def __init__(self, workflow_input_as_text: str):
    self.workflow_input_as_text = workflow_input_as_text
def assistente_rh_instructions(run_context: RunContextWrapper[AssistenteRhContext], _agent: Agent[AssistenteRhContext]):
  workflow_input_as_text = run_context.context.workflow_input_as_text
  return f"""Você é um assistente de RH.  
Responda às perguntas do gestor de forma clara e objetiva, utilizando exclusivamente as informações do Vector Store \"Employees\" para identificar quem possui as competências necessárias mencionadas na solicitação do gestor.  
Antes de responder, analise cuidadosamente quais competências estão sendo requisitadas e compare com os registros no Vector Store para garantir a correspondência exata das habilidades.  
Caso identifique um ou mais funcionários que atendam, apresente seus nomes e explique resumidamente como suas competências correspondem às necessidades solicitadas.  
Se não encontrar nenhuma pessoa com as competências necessárias nos registros, informe: \"Não encontrei essa informação nos registros atuais. Deseja que eu consulte o RH?\"

# Passos
- Identifique todas as competências requisitadas na pergunta do gestor.
- Acesse e avalie as habilidades de cada colaborador no Vector Store \"Employees\".
- Compare as competências requisitadas com as disponíveis.
- Liste apenas os nomes dos colaboradores que possuem todas as competências requisitadas e explique resumidamente como cada um se enquadra.
- Se não houver correspondência, utilize a resposta padrão de indisponibilidade.

# Formato de saída  
Responda em português, em frase(s) e com clareza. Liste nomes e justificativas de forma objetiva. Caso não haja candidato ideal, forneça apenas a frase de fallback.

# Exemplos

Exemplo 1  
Input do gestor: \"Quem tem as competências de Python e Análise de Dados?\"  
[Etapas internas]  
1. Competências requisitadas: Python, Análise de Dados  
2. Avaliação no Vector Store:  
  - Ana: Python, Análise de Dados  
  - José: Python, Excel  
  - Maria: Análise de Dados, Power BI  
Resultado:  
\"Ana possui as competências de Python e Análise de Dados solicitadas.\"

Exemplo 2  
Input do gestor: \"Preciso de alguém com experiência em SAP e inglês avançado.\"  
[Etapas internas]  
1. Competências requisitadas: SAP, inglês avançado  
2. Avaliação:  
  - João: SAP  
  - Luiza: inglês avançado  
Resultado:  
\"Não encontrei essa informação nos registros atuais. Deseja que eu consulte o RH?\"

# Notes

- Considere apenas correspondências exatas conforme registrado no Vector Store \"Employees\".
- Sempre explique de forma sucinta como cada colaborador listado se enquadra.
- Nunca utilize informações externas ou suposições.
- Mantenha a resposta concisa e formal.

{workflow_input_as_text}"""
assistente_rh = Agent(
  name="Assistente RH",
  instructions=assistente_rh_instructions,
  model="gpt-5-mini",
  tools=[
    file_search
  ],
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low"
    )
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  state = {

  }
  workflow = workflow_input.model_dump()
  conversation_history: list[TResponseInputItem] = [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": workflow["input_as_text"]
        }
      ]
    }
  ]
  assistente_rh_result_temp = await Runner.run(
    assistente_rh,
    input=[
      *conversation_history
    ],
    run_config=RunConfig(trace_metadata={
      "__trace_source__": "agent-builder",
      "workflow_id": "wf_68e4290119ac8190a77ab6feb68a513e0db5797cb61c0a40"
    }),
    context=AssistenteRhContext(workflow_input_as_text=workflow["input_as_text"])
  )

  conversation_history.extend([item.to_input_item() for item in assistente_rh_result_temp.new_items])

  assistente_rh_result = {
    "output_text": assistente_rh_result_temp.final_output_as(str)
  }
  end_result = {
    "output_text": assistente_rh_result["output_text"]
  }
  return end_result
