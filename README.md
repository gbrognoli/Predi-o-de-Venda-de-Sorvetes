📊 Análise e Previsão de Vendas de Sorvetes 🍦
Este projeto realiza uma análise exploratória e modelagem preditiva de dados de vendas de sorvetes com base na temperatura. Inclui visualizações interativas, machine learning com árvores de decisão e geração de insights acionáveis.

🛠️ Funcionalidades
Análise Exploratória de Dados:

Relação entre temperatura e vendas de sorvetes.
Resumo das vendas por temperatura.
Mapa de calor das vendas por loja e temperatura.
Análise de pontos de venda com melhor e pior desempenho.
Modelagem Preditiva (Machine Learning):

Previsão de vendas futuras usando uma árvore de decisão.
Simulação de temperaturas futuras para os próximos 3 meses.
Visualização das previsões de vendas ao longo do tempo.
Visualizações Interativas:

Gráficos interativos para explorar dados e previsões.
Apresentação consolidada em HTML para fácil compartilhamento.
📂 Estrutura do Projeto
plaintext
Copiar código
.
├── data/
│   ├── base de dados                # Arquivo de dados de entrada
│   ├── previsoes_futuras.csv        # Arquivo com previsões futuras
├── visuals/
│   ├── presentation_vendas_sorvetes.html  # Apresentação interativa
├── src/
│   ├── main_analysis.py             # Código principal do projeto
├── README.md                        # Documentação do projeto

🚀 Tecnologias Utilizadas
Linguagem: Python
Bibliotecas Principais:
pandas e numpy para manipulação de dados.
matplotlib e seaborn para visualizações estáticas.
plotly para visualizações interativas.
scikit-learn para modelagem preditiva.
📈 Fluxo de Trabalho
Carregar os Dados:

O projeto começa importando um arquivo Excel com dados históricos de temperatura e vendas por ponto de venda.
Análise Exploratória:

Análise da relação entre temperatura e vendas.
Identificação de padrões e tendências usando gráficos e mapas de calor.
Treinamento do Modelo de Machine Learning:

Divisão dos dados em conjuntos de treinamento e teste.
Treinamento de uma árvore de decisão para prever vendas futuras com base na temperatura.
Previsões Futuras:

Geração de previsões para os próximos 3 meses simulando temperaturas futuras.
Exportação das previsões para um arquivo CSV.
Apresentação Consolidada:

Criação de uma apresentação HTML com visualizações interativas.
🧪 Como Executar o Projeto
Pré-requisitos
Python 3.7+
Bibliotecas Python: instale com pip install -r requirements.txt
Passos
Clone este repositório:

git clone https://github.com/seu-usuario/sorvete-vendas-predicao.git
cd sorvete-vendas-predicao
Certifique-se de que o arquivo de dados está localizado na pasta data/.

Execute o script principal:
python src/main_analysis.py
Verifique os resultados:

Previsões: Arquivo previsoes_futuras.csv gerado na pasta data/.
Apresentação Interativa: Arquivo presentation_vendas_sorvetes.html na pasta visuals/

📊 Exemplos de Resultados
Relação entre Temperatura e Vendas:

Mapa de Calor de Vendas:

Previsão de Vendas Futura:

🤝 Contribuições
Contribuições são bem-vindas! Siga os passos abaixo para contribuir:

Faça um fork do repositório.
Crie uma nova branch: git checkout -b minha-feature.
Commit suas alterações: git commit -m 'Minha nova feature'.
Faça o push da sua branch: git push origin minha-feature.
Abra um pull request.

Autor
Gabriel Brognoli
