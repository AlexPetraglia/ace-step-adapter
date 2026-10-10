# ace-step-adapter

Um serviço HTTP local que envolve o [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) e expõe **o mesmo contrato da API REST do ACE-Step** para as rotas usadas por clientes como a CLI do [gerador-musical](https://github.com/AlexPetraglia/gerador-musical).

> **Status:** em desenvolvimento. Esta versão do README descreve o objetivo, a arquitetura e o contrato planejado. Nada aqui deve ser lido como funcionalidade já validada.

---

## Por que existe

O ACE-Step 1.5 roda em GPUs com pouca memória, mas em uma RTX 3050 de 6 GB a geração de áudio não completava. Os testes mostraram que:

- a etapa de difusão (DiT) termina em poucos segundos;
- a decodificação do áudio (VAE) trava quando a VRAM livre fica perto de zero;
- o modelo principal fica retido na GPU durante toda a sessão (`Keeping main model on cuda (persistent)`).

Com **quantização INT8 dos pesos do DiT** e **descarregamento para a CPU entre as etapas** (`offload_to_cpu=True`), a geração de 20 s completou em cerca de 1,3 s de decodificação, com áudio válido. Essas opções não estão disponíveis na API do ACE-Step, então o adapter as aplica diretamente no handler Python do ACE-Step.

O adapter existe para:

1. **Isolar a GPU** em um processo próprio, com seus logs e seu timeout, separado do cliente.
2. **Manter o contrato da API do ACE-Step**, para que um cliente que já fala com o ACE-Step possa usar o adapter sem mudanças.
3. **Facilitar a troca de ambiente.** Se você passar a usar o ACE-Step oficial, em outra máquina ou em uma VPS, basta apontar o cliente para a nova URL.

---

## Arquitetura

```
┌──────────────┐   HTTP (contrato ACE-Step)   ┌─────────────────────┐   Python   ┌──────────────────┐
│    Cliente   │ ───────────────────────────▶ │  ace-step-adapter   │ ─────────▶ │ handler ACE-Step │
│ (ex.: CLI)   │ ◀─────────────────────────── │  (FastAPI, GPU)     │ ◀───────── │ (DiT, VAE, ...)  │
└──────────────┘                              └─────────────────────┘            └──────────────────┘
```

O ACE-Step continua instalado e intocado, em sua própria pasta. O adapter importa o pacote `acestep` como biblioteca e controla a inicialização e a geração.

---

## Contrato implementado

A versão inicial implementa apenas as rotas que um cliente de geração precisa. O fluxo é assíncrono: envia-se uma tarefa, consulta-se o status até a conclusão e baixa-se o arquivo gerado.

| Método | Rota | Função |
|--------|------|--------|
| `GET`  | `/health` | Verifica se o serviço está ativo e com o modelo carregado. |
| `POST` | `/release_task` | Cria uma tarefa de geração e devolve o `task_id`. |
| `POST` | `/query_result` | Consulta o status de uma ou mais tarefas (`task_id_list`). |
| `GET`  | `/v1/audio?path=...` | Baixa o arquivo gerado. Só serve arquivos da pasta de saída do serviço. |

Formato das respostas: `{"data": ..., "code": 200, "error": null, "timestamp": ..., "extra": null}`. No resultado de `/query_result`, o campo `result` é uma string JSON, como na API do ACE-Step.

Status das tarefas: `0` em andamento, `1` concluída, `2` falhou.

Outras rotas da API do ACE-Step (treinamento, formatação de letra, modelos, etc.) **não** fazem parte do escopo inicial.

---

## Configuração (planejada)

Os parâmetros de execução ficam em um arquivo de configuração do próprio adapter, não em variáveis de ambiente do servidor:

- caminho da instalação do ACE-Step;
- modelo DiT (padrão: `acestep-v15-turbo`);
- quantização do DiT (padrão: `int8_weight_only`);
- `offload_to_cpu` (padrão: `true`);
- pasta de saída dos arquivos;
- porta e endereço de escuta (padrão: `127.0.0.1`).

---

## Requisitos

- Windows 10/11 ou Linux com GPU NVIDIA e drivers atualizados.
- Python compatível com o ACE-Step 1.5 (3.11 ou 3.12).
- ACE-Step 1.5 instalado e com os modelos baixados, em uma pasta separada deste repositório.
- Para quantização INT8: `torchao`, já presente no ambiente do ACE-Step.

---

## Limitações conhecidas

- Testado apenas em uma máquina com RTX 3050 Laptop de 6 GB. Outras configurações não foram validadas.
- Uma geração por vez. A GPU não é compartilhada entre tarefas.
- O adapter não é um proxy: ele executa o ACE-Step diretamente. Por isso, um travamento dentro do ACE-Step ainda pode bloquear a GPU. O adapter aplica timeout nas tarefas, mas não consegue interromper uma operação CUDA em andamento.
- Dependência de versão: o adapter usa a API Python interna do ACE-Step, que pode mudar entre versões. A versão testada será indicada na documentação de instalação.

---

## Instalação

> Em preparação. A instalação, o comando de execução e os testes serão documentados quando a implementação estiver pronta.

---

## Relação com o ACE-Step

Este projeto é independente e não é afiliado ao projeto ACE-Step. Ele depende do ACE-Step 1.5, que é distribuído sob licença MIT. Os modelos têm licenças próprias, verifique-as antes de usar.

---

## Licença

A definir.
