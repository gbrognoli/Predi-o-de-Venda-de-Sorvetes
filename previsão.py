import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ======================== Gerar Base de Dados ========================

# Criar 100 dias de dados simulados
np.random.seed(42)
dias = np.arange(1, 101)  # Dias de 1 a 100
temperaturas = np.random.uniform(15, 35, size=100)  # Temperaturas entre 15°C e 35°C
vendas = (temperaturas * np.random.uniform(20, 30, size=100) + np.random.normal(0, 50, size=100)).astype(int)

# Simular vendas por 10 pontos de venda
pontos_de_venda = [f"V{i}" for i in range(1, 11)]
vendas_por_ponto = {ponto: (vendas + np.random.normal(0, 20, size=100)).astype(int) for ponto in pontos_de_venda}

# Criar DataFrame final
dados = pd.DataFrame({'Dia': dias, 'Temp': temperaturas, **vendas_por_ponto})

# ======================== Análise Exploratória ========================

# 1. Relação entre Temperatura e Vendas de Sorvetes
dados['Total_Vendas'] = dados.iloc[:, 2:].sum(axis=1)

plt.figure(figsize=(10, 6))
plt.scatter(dados['Temp'], dados['Total_Vendas'], alpha=0.7)
plt.title('Relação entre Temperatura e Vendas de Sorvetes')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Total de Vendas')
plt.grid(True)
plt.show()

# 2. Resumo das Vendas por Temperatura
df_summary = dados[['Temp', 'Total_Vendas']]

# 3. Mapa de Calor das Vendas de Sorvetes
vendas_data = dados.iloc[:, 2:-1]
plt.figure(figsize=(12, 8))
sns.heatmap(vendas_data, annot=False, fmt=".0f", cmap="YlGnBu", cbar=True, linewidths=.5)
plt.title('Mapa de Calor das Vendas de Sorvetes')
plt.xlabel('Pontos de Venda')
plt.ylabel('Dias')
plt.show()

# 4. Vendas por Ponto de Venda e Temperatura
vendas_por_ponto = dados.set_index('Dia').iloc[:, 1:-1].transpose()
plt.figure(figsize=(12, 8))
for ponto in vendas_por_ponto.columns:
    plt.plot(vendas_por_ponto.index, vendas_por_ponto[ponto], marker='o', label=f'Dia {ponto}')
plt.title('Vendas por Ponto de Venda e Temperatura')
plt.xlabel('Pontos de Venda')
plt.ylabel('Vendas')
plt.legend(title='Dia')
plt.grid(True)
plt.show()

# ======================== Machine Learning ========================

# Ajustar X e y para garantir alinhamento correto
X = pd.DataFrame({'Temp': dados['Temp'].repeat(len(pontos_de_venda))})
y = dados.iloc[:, 2:-1].values.flatten()

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de árvore de decisão
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Avaliar o desempenho do modelo
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Erro Médio Absoluto (MAE): {mae}")
print(f"R² Score: {r2}")

# Prever as vendas ao longo de 3 meses simulados
future_temps = np.linspace(15, 35, 90)  # 3 meses (90 dias)
future_df = pd.DataFrame({'Temp': future_temps})
future_predictions = model.predict(future_df)

# Visualizar previsões futuras
plt.figure(figsize=(12, 6))
plt.plot(future_temps, future_predictions, label="Previsões Futuras", color="blue")
plt.title("Previsão de Vendas por Temperatura ao Longo de 3 Meses")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Previsão de Vendas")
plt.legend()
plt.grid(True)
plt.show()

# Salvar previsões futuras em um arquivo
future_df['Predicted_Sales'] = future_predictions
future_df.to_csv('previsoes_futuras.csv', index=False)

# ======================== Apresentação Interativa ========================

fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "Relação Entre Temperatura e Vendas de Sorvetes",
        "Resumo das Vendas por Temperatura",
        "Mapa de Calor das Vendas de Sorvetes",
        "Vendas por Ponto de Venda e Temperatura",
        "Top-Performing Sales Points",
        "Low-Performing Sales Points",
        "Vendas a 30°C por Ponto de Venda",
        "Vendas Ordenadas a 15°C por Ponto de Venda"
    )
)

# Adicionar gráficos ao subplot
fig.add_trace(go.Scatter(x=dados['Temp'], y=dados['Total_Vendas'], mode='markers+lines', name='Vendas Totais'), row=1, col=1)
fig.add_trace(go.Bar(x=dados['Temp'], y=dados['Total_Vendas'], name='Vendas por Temperatura'), row=1, col=2)
heatmap_data = go.Heatmap(z=vendas_data.values, x=vendas_data.columns, y=dados['Dia'], colorscale='Viridis')
fig.add_trace(heatmap_data, row=2, col=1)
for ponto in vendas_por_ponto.columns:
    fig.add_trace(go.Scatter(x=vendas_por_ponto.index, y=vendas_por_ponto[ponto], mode='lines+markers', name=f'Dia {ponto}'), row=2, col=2)

# Layout geral
fig.update_layout(height=1000, width=1500, title_text="Análise de Vendas de Sorvetes: Apresentação Completa", showlegend=False)

# Salvar apresentação como HTML
presentation_path = "presentation_vendas_sorvetes.html"
fig.write_html(presentation_path)

