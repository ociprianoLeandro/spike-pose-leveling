# Spike Pose Leveling

Projeto de nivelamento desenvolvido para introduzir os fundamentos de visão computacional que servirão de base para o **Spike.AI**. O objetivo principal é receber um vídeo de um atleta de voleibol, extrair a pose corporal por meio do MediaPipe Pose e renderizar apenas o esqueleto digital sobre um fundo preto, descartando totalmente a imagem original.

## Estrutura do Projeto

O código foi dividido em módulos para manter cada responsabilidade isolada:

```text
spike-pose-leveling/
│
├── input/
│   └── nome.mp4          # Vídeo original de entrada
├── output/
│   └── nome_pose.mp4     # Vídeo final gerado com o skeleton
├── models/
│   └── pose_landmarker_lite.task  # Modelo leve do MediaPipe
├── src/
│   ├── main.py             # Orquestra o fluxo de execução e métricas
│   ├── video.py            # Leitura e escrita de vídeo via OpenCV
│   ├── pose.py             # Detecção e extração dos 33 landmarks corporais
│   └── renderer.py         # Criação da tela preta e desenho de linhas/pontos
├── requirements.txt
├── .gitignore
└── README.md
```

## Tecnologias Utilizadas

* **Python 3.11+**
* **OpenCV (opencv-python-headless):** captura de frames, manipulação de matrizes de imagem e escrita do arquivo final.
* **MediaPipe 1.0+:** detecção e rastreamento dos 33 pontos anatômicos.
* **NumPy:** criação dos frames pretos e gerenciamento de buffers visuais.

## Como Instalar e Executar

### 1. Clonar o repositório e preparar o ambiente

No terminal (Linux / WSL / macOS):

```bash
# Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd spike-pose-leveling

# Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Executar o processamento

Coloque o vídeo desejado na pasta `input/` e execute:

```bash
python3 src/main.py --input input/nome.mp4 --output output/nome_pose.mp4
```

Parâmetros disponíveis:

* `--input`: caminho do arquivo de vídeo obrigatório.
* `--output`: caminho de saída do vídeo processado (padrão: `output/video_pose.mp4`).
* `--visibility`: nível de confiança mínimo para desenhar um landmark (padrão: `0.5`).

## Decisões de Implementação e Motivações

* **Modularização direta:** em vez de usar classes complexas e abstrações pesadas, optei por funções claras em cada arquivo. `video.py` cuida apenas do OpenCV, `pose.py` isola a inferência do modelo e `renderer.py` cuida da geometria na tela.
* **Manipulação matemática das coordenadas:** optei por não usar funções prontas de desenho da biblioteca (`mp_drawing`). O código itera diretamente sobre as coordenadas normalizadas $(x, y)$, converte para pixels multiplicando pelas dimensões do frame e desenha cada reta e círculo via OpenCV. Isso garante controle total sobre os dados numéricos para a futura fase de cálculo de ângulos e biomecânica do Spike.AI.
* **Tratamento de frames vazios:** quando o atleta sai do enquadramento ou o MediaPipe não detecta ninguém, o sistema gera o frame preto vazio e continua a gravação normalmente, sem quebras de execução.
* **Filtro por confiança (Visibility):** implementado para evitar artefatos visuais quando membros sofrem oclusão durante o salto ou movimento de ataque.

## Dificuldades e Limitações Encontradas