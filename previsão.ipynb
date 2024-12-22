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

# ======================== Carregar os Dados ========================
# Carregar dados do arquivo Excel
file_path = 'sorvete c temperatura.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse('Plan1')

# Criar a coluna Total_Vendas (soma das vendas por temperatura)
df['Total_Vendas'] = df.iloc[:, 1:].sum(axis=1)

# ======================== Análise Exploratória ========================

# 1. Relação entre Temperatura e Vendas de Sorvetes
plt.figure(figsize=(10, 6))
plt.scatter(df['Temp'], df['Total_Vendas'], alpha=0.7)
plt.title('Relação entre Temperatura e Vendas de Sorvetes')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Total de Vendas')
plt.grid(True)
plt.show()

# 2. Resumo das Vendas por Temperatura
df_summary = df[['Temp', 'Total_Vendas']]

# 3. Mapa de Calor das Vendas de Sorvetes
vendas_data = df.iloc[:, 1:-1]
plt.figure(figsize=(12, 8))
sns.heatmap(vendas_data, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True, linewidths=.5)
plt.title('Mapa de Calor das Vendas de Sorvetes')
plt.xlabel('Pontos de Venda')
plt.ylabel('Temperatura (°C)')
plt.show()

# 4. Vendas por Ponto de Venda e Temperatura
vendas_por_ponto = df.set_index('Temp').iloc[:, :-1].transpose()
plt.figure(figsize=(12, 8))
for ponto in vendas_por_ponto.columns:
    plt.plot(vendas_por_ponto.index, vendas_por_ponto[ponto], marker='o', label=f'Temp {ponto}°C')
plt.title('Vendas por Ponto de Venda e Temperatura')
plt.xlabel('Pontos de Venda')
plt.ylabel('Vendas')
plt.legend(title='Temperatura (°C)')
plt.grid(True)
plt.show()

# 5. Top-Performing Sales Points
sales_by_point = df.iloc[:, 1:-1].sum()
top_sales_points = sales_by_point.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
top_sales_points.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Top-Performing Sales Points')
plt.xlabel('Sales Points')
plt.ylabel('Total Sales')
plt.grid(axis='y')
plt.show()

# 6. Low-Performing Sales Points
low_sales_points = sales_by_point.sort_values(ascending=True)
plt.figure(figsize=(12, 8))
low_sales_points.plot(kind='bar', color='salmon', alpha=0.8)
plt.title('Low-Performing Sales Points')
plt.xlabel('Sales Points')
plt.ylabel('Total Sales')
plt.grid(axis='y')
plt.show()

# 7. Vendas a 30°C
hot_temp_sales = df[df['Temp'] == 30].iloc[:, 1:-1].sum()
plt.figure(figsize=(12, 8))
hot_temp_sales.plot(kind='bar', color='orange', alpha=0.8)
plt.title('Vendas a 30°C por Ponto de Venda')
plt.xlabel('Pontos de Venda')
plt.ylabel('Total de Vendas a 30°C')
plt.grid(axis='y')
plt.show()

# 8. Vendas a 15°C
sales_at_15 = df[df['Temp'] == 15].iloc[:, 1:-1].sum()
sorted_sales_at_15 = sales_at_15.sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sorted_sales_at_15.plot(kind='bar', color='teal', alpha=0.8)
plt.title('Vendas Ordenadas a 15°C por Ponto de Venda')
plt.xlabel('Pontos de Venda')
plt.ylabel('Total de Vendas a 15°C')
plt.grid(axis='y')
plt.show()

# ======================== Machine Learning ========================

# Ajustar X e y para garantir alinhamento correto
X = pd.DataFrame({'Temp': df['Temp'].repeat(df.iloc[:, 1:-1].shape[1])})
y = df.iloc[:, 1:-1].values.flatten()

# Verificar os tamanhos ajustados
print(f"X shape: {X.shape}, y shape: {y.shape}")

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
fig.add_trace(go.Scatter(x=df['Temp'], y=df['Total_Vendas'], mode='markers+lines', name='Vendas Totais'), row=1, col=1)
fig.add_trace(go.Bar(x=df['Temp'], y=df['Total_Vendas'], name='Vendas por Temperatura'), row=1, col=2)
heatmap_data = go.Heatmap(z=vendas_data.values, x=vendas_data.columns, y=df['Temp'], colorscale='Viridis')
fig.add_trace(heatmap_data, row=2, col=1)
for ponto in vendas_por_ponto.columns:
    fig.add_trace(go.Scatter(x=vendas_por_ponto.index, y=vendas_por_ponto[ponto], mode='lines+markers', name=f'Temp {ponto}°C'), row=2, col=2)
fig.add_trace(go.Bar(x=top_sales_points.index, y=top_sales_points.values, name='Top Sales Points'), row=3, col=1)
fig.add_trace(go.Bar(x=low_sales_points.index, y=low_sales_points.values, name='Low Sales Points'), row=3, col=2)
fig.add_trace(go.Bar(x=hot_temp_sales.index, y=hot_temp_sales.values, name='Vendas a 30°C'), row=4, col=1)
fig.add_trace(go.Bar(x=sorted_sales_at_15.index, y=sorted_sales_at_15.values, name='Vendas a 15°C'), row=4, col=2)

# Layout geral
fig.update_layout(height=1000, width=1500, title_text="Análise de Vendas de Sorvetes: Apresentação Completa", showlegend=False)

# Salvar apresentação como HTML
presentation_path = "presentation_vendas_sorvetes.html"
fig.write_html(presentation_path)
