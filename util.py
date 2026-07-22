"""
Funções auxiliares para o projeto de detecção de fraudes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import datetime
from math import radians, sin, cos, sqrt, atan2
import warnings


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
warnings.filterwarnings('ignore')


# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_data(file_path):
    """
    Carrega os dados do arquivo CSV
    
    Args:
        file_path (str): Caminho para o arquivo CSV
        
    Returns:
        pandas.DataFrame: DataFrame com os dados carregados
    """
    return pd.read_csv(file_path)

def plot_distribution(df, column, title, figsize=(12, 6)):
    """
    Plota a distribuição de uma variável
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        column (str): Nome da coluna a ser plotada
        title (str): Título do gráfico
        figsize (tuple): Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    if df[column].dtype == 'object' or df[column].nunique() < 20:
        # Para variáveis categóricas
        counts = df[column].value_counts().sort_values(ascending=False)
        if len(counts) > 15:
            # Se houver muitas categorias, mostrar apenas as 15 mais frequentes
            counts = counts.head(15)
            title += " (15 categorias mais frequentes)"
        
        ax = sns.barplot(x=counts.index, y=counts.values)
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.ylabel('Contagem')
        plt.tight_layout()
    else:
        # Para variáveis numéricas
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Histograma - {title}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot - {title}')
        
        plt.tight_layout()
    
    plt.show()

def plot_target_distribution(df, target='is_fraud', figsize=(10, 6)):
    """
    Plota a distribuição da variável alvo
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        target (str): Nome da coluna alvo
        figsize (tuple): Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    # Contagem de cada classe
    counts = df[target].value_counts()
    
    # Calcular a proporção
    total = len(df)
    proportions = counts / total * 100
    
    # Criar o gráfico de barras
    ax = sns.barplot(x=counts.index, y=counts.values)
    
    # Adicionar rótulos com as contagens e proporções
    for i, (count, prop) in enumerate(zip(counts, proportions)):
        ax.text(i, count/2, f'{count}\n({prop:.2f}%)', 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Configurar o gráfico
    plt.title(f'Distribuição da variável {target}')
    plt.xlabel(target)
    plt.ylabel('Contagem')
    plt.xticks([0, 1], ['Não Fraude (0)', 'Fraude (1)'])
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir informações adicionais
    print(f"Total de registros: {total}")
    print(f"Registros não fraudulentos: {counts[0]} ({proportions[0]:.2f}%)")
    print(f"Registros fraudulentos: {counts[1]} ({proportions[1]:.2f}%)")
    print(f"Proporção de desbalanceamento: 1:{counts[0]/counts[1]:.2f}")

def plot_categorical_vs_target(df, column, target='is_fraud', top_n=10, figsize=(14, 7)):
    """
    Plota a relação entre uma variável categórica e a variável alvo
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        column (str): Nome da coluna categórica
        target (str): Nome da coluna alvo
        top_n (int): Número de categorias a serem mostradas
        figsize (tuple): Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    # Calcular a taxa de fraude por categoria
    fraud_rate = df.groupby(column)[target].mean().sort_values(ascending=False)
    
    # Calcular a contagem por categoria
    counts = df[column].value_counts()
    
    # Filtrar para mostrar apenas as top_n categorias mais frequentes
    top_categories = counts.head(top_n).index
    
    # Filtrar o DataFrame para incluir apenas as top_n categorias
    df_filtered = df[df[column].isin(top_categories)]
    
    # Criar o gráfico
    plt.subplot(1, 2, 1)
    sns.countplot(x=column, hue=target, data=df_filtered, order=counts.head(top_n).index)
    plt.title(f'Contagem de {column} por {target} (Top {top_n})')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Não Fraude', 'Fraude'])
    
    plt.subplot(1, 2, 2)
    fraud_rate_filtered = fraud_rate[fraud_rate.index.isin(top_categories)]
    sns.barplot(x=fraud_rate_filtered.index, y=fraud_rate_filtered.values, order=fraud_rate_filtered.sort_values(ascending=False).index)
    plt.title(f'Taxa de Fraude por {column} (Top {top_n})')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Taxa de Fraude')
    
    plt.tight_layout()
    plt.show()

def plot_numerical_vs_target(df, column, target='is_fraud', figsize=(14, 6)):
    """
    Plota a relação entre uma variável numérica e a variável alvo
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        column (str): Nome da coluna numérica
        target (str): Nome da coluna alvo
        figsize (tuple): Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=target, y=column, data=df)
    plt.title(f'Boxplot de {column} por {target}')
    plt.xticks([0, 1], ['Não Fraude', 'Fraude'])
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=column, hue=target, kde=True, element="step", common_norm=False)
    plt.title(f'Distribuição de {column} por {target}')
    plt.legend(['Não Fraude', 'Fraude'])
    
    plt.tight_layout()
    plt.show()

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula a distância Haversine entre dois pontos na Terra
    
    Args:
        lat1 (float): Latitude do ponto 1
        lon1 (float): Longitude do ponto 1
        lat2 (float): Latitude do ponto 2
        lon2 (float): Longitude do ponto 2
        
    Returns:
        float: Distância em quilômetros
    """
    # Converter graus para radianos
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Raio da Terra em km
    
    return distance

def extract_first_digit(number):
    """
    Extrai o primeiro dígito de um número
    
    Args:
        number (float): Número
        
    Returns:
        int: Primeiro dígito
    """
    # Converter para string e pegar o primeiro caractere
    if pd.isna(number) or number == 0:
        return 0
    
    # Remover sinal negativo se existir
    str_num = str(abs(number))
    
    # Encontrar o primeiro dígito não-zero
    for char in str_num:
        if char.isdigit() and char != '0' and char != '.':
            return int(char)
    
    return 0

def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plota a matriz de correlação
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
        figsize (tuple): Tamanho da figura
    """
    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calcular a matriz de correlação
    corr_matrix = numeric_df.corr()
    
    # Plotar a matriz de correlação
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 8)):
    """
    Plota a importância das features para um modelo
    
    Args:
        model: Modelo treinado (deve ter o atributo feature_importances_)
        feature_names (list): Lista com os nomes das features
        top_n (int): Número de features a serem mostradas
        figsize (tuple): Tamanho da figura
    """
    # Verificar se o modelo tem o atributo feature_importances_
    if not hasattr(model, 'feature_importances_'):
        print("O modelo não possui o atributo feature_importances_")
        return
    
    # Obter a importância das features
    importances = model.feature_importances_
    
    # Criar um DataFrame com as importâncias
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Ordenar por importância
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Selecionar as top_n features
    if len(feature_importance) > top_n:
        feature_importance = feature_importance.head(top_n)
    
    # Plotar
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top {top_n} Features mais Importantes')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Avalia o modelo com várias métricas
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        threshold (float): Limiar de classificação
        
    Returns:
        dict: Dicionário com as métricas
    """
    # Fazer predições
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calcular métricas
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Precision at K
    # Ordenar as probabilidades em ordem decrescente
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_test_sorted = y_test.iloc[sorted_indices] if hasattr(y_test, 'iloc') else y_test[sorted_indices]
    
    # Calcular precision at 100
    k = 100
    precision_at_k = np.sum(y_test_sorted[:k]) / k
    
    # Resultados
    results = {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'average_precision': avg_precision,
        'precision_at_100': precision_at_k
    }
    
    return results

def plot_evaluation_metrics(y_test, y_pred_proba, figsize=(16, 12)):
    """
    Plota as curvas ROC e Precision-Recall
    
    Args:
        y_test: Target de teste
        y_pred_proba: Probabilidades preditas
        figsize (tuple): Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    # ROC Curve
    plt.subplot(2, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    
    # Precision-Recall Curve
    plt.subplot(2, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.plot(recall, precision, label=f'AUC-PR = {pr_auc:.4f}\nAP = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='upper right')
    
    # Precision at K
    plt.subplot(2, 2, 3)
    # Ordenar as probabilidades em ordem decrescente
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_test_sorted = y_test.iloc[sorted_indices] if hasattr(y_test, 'iloc') else y_test[sorted_indices]
    
    # Calcular precision at k para diferentes valores de k
    ks = [10, 50, 100, 500, 1000]
    precision_at_ks = []
    
    for k in ks:
        if k <= len(y_test_sorted):
            precision_at_k = np.sum(y_test_sorted[:k]) / k
            precision_at_ks.append(precision_at_k)
        else:
            precision_at_ks.append(None)
    
    # Plotar precision at k
    plt.bar(range(len(ks)), precision_at_ks)
    plt.xticks(range(len(ks)), [str(k) for k in ks])
    plt.xlabel('K')
    plt.ylabel('Precision at K')
    plt.title('Precision at K')
    
    # Distribuição das probabilidades
    plt.subplot(2, 2, 4)
    sns.histplot(y_pred_proba, bins=50, kde=True)
    plt.xlabel('Probabilidade Predita')
    plt.ylabel('Contagem')
    plt.title('Distribuição das Probabilidades Preditas')
    
    plt.tight_layout()
    plt.show()

def plot_threshold_analysis(y_test, y_pred_proba, cost_fp=1, cost_fn=10, figsize=(16, 12)):
    """
    Plota a análise de threshold com base nos custos
    
    Args:
        y_test: Target de teste
        y_pred_proba: Probabilidades preditas
        cost_fp (float): Custo de um falso positivo
        cost_fn (float): Custo de um falso negativo
        figsize (tuple): Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    # Calcular métricas para diferentes thresholds
    thresholds = np.linspace(0, 1, 100)
    metrics = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calcular custo total
        total_cost = fp * cost_fp + fn * cost_fn
        
        metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp': fp,
            'fn': fn,
            'total_cost': total_cost
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Plotar métricas vs threshold
    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
    plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Valor')
    plt.title('Métricas vs Threshold')
    plt.legend()
    plt.grid(True)
    
    # Plotar custo total vs threshold
    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['threshold'], metrics_df['total_cost'])
    
    # Encontrar o threshold ótimo (menor custo)
    optimal_idx = metrics_df['total_cost'].idxmin()
    optimal_threshold = metrics_df.loc[optimal_idx, 'threshold']
    optimal_cost = metrics_df.loc[optimal_idx, 'total_cost']
    
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Threshold Ótimo = {optimal_threshold:.2f}')
    plt.axhline(y=optimal_cost, color='g', linestyle='--', 
                label=f'Custo Mínimo = {optimal_cost:.2f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Custo Total')
    plt.title(f'Custo Total vs Threshold (FP={cost_fp}, FN={cost_fn})')
    plt.legend()
    plt.grid(True)
    
    # Plotar FP e FN vs threshold
    plt.subplot(2, 2, 3)
    plt.plot(metrics_df['threshold'], metrics_df['fp'], label='Falsos Positivos')
    plt.plot(metrics_df['threshold'], metrics_df['fn'], label='Falsos Negativos')
    plt.xlabel('Threshold')
    plt.ylabel('Contagem')
    plt.title('Falsos Positivos e Falsos Negativos vs Threshold')
    plt.legend()
    plt.grid(True)
    
    # Comparar threshold ótimo vs threshold padrão (0.5)
    plt.subplot(2, 2, 4)
    
    # Threshold padrão
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    tn_default, fp_default, fn_default, tp_default = confusion_matrix(y_test, y_pred_default).ravel()
    default_cost = fp_default * cost_fp + fn_default * cost_fn
    
    # Threshold ótimo
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    tn_optimal, fp_optimal, fn_optimal, tp_optimal = confusion_matrix(y_test, y_pred_optimal).ravel()
    optimal_cost = fp_optimal * cost_fp + fn_optimal * cost_fn
    
    # Plotar comparação
    labels = ['Threshold = 0.5', f'Threshold = {optimal_threshold:.2f}']
    costs = [default_cost, optimal_cost]
    
    plt.bar(labels, costs)
    plt.ylabel('Custo Total')
    plt.title('Comparação de Custos: Threshold Padrão vs Ótimo')
    
    # Adicionar valores nos topos das barras
    for i, cost in enumerate(costs):
        plt.text(i, cost + 0.05 * max(costs), f'{cost:.2f}', 
                 ha='center', va='bottom')
    
    # Calcular a economia
    savings = default_cost - optimal_cost
    savings_percent = (savings / default_cost) * 100
    
    plt.figtext(0.5, 0.01, 
                f'Economia com threshold ótimo: {savings:.2f} ({savings_percent:.2f}%)',
                ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return optimal_threshold, optimal_cost

def time_plot(serie: pd.Series, c='Green', stats=True):
    """
    Função de criação automatica de séries temporais e análise.
    :param serie: Série que origininará o gráfico
    :param c: Cor do gráfico
    :param stats: Indicador se traz ou não as estatísticas/plots estatísticos da série temporal
    :return: None
    """

    # Garantir que a série esteja em datetime
    if serie.dtype != 'datetime64[ns]':
        serie = serie.astype(str)
        serie = pd.to_datetime(serie, format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Contabilizando Na's e mostrando o range
    print('Número de missings: {}'.format(pd.isna(serie).sum()))
    print('Range da série: {} - {}'.format(serie.min().strftime('%d/%b/%Y'),
                                           serie.max().strftime('%d/%b/%Y')))

    dias = (serie.max() - serie.min()).days
    if dias // 365 == 1:
        print('\t- {}Ano {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))
    else:
        print('\t- {}Anos {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))

    # Destrinchando a série em dia, mês e ano
    serie_dia = serie.dt.strftime('%d/%m/%Y').copy()
    serie_mes = serie.dt.strftime('%b/%Y').copy()
    serie_ano = serie.dt.strftime('%Y').copy()

    # Para os dados diários:
    df_dia = serie_dia.value_counts().rename(serie.name)
    df_dia.index.name = "day"  # define um nome diferente para o índice
    df_dia = df_dia.reset_index()
    df_dia['date'] = pd.to_datetime(df_dia['day'], format='%d/%m/%Y')
    df_dia['media movel (30)'] = df_dia[serie.name].rolling(window=30).mean()
    df_dia.sort_values('date', inplace=True)

    # Para os dados mensais:
    df_mes = serie_mes.value_counts().rename(serie.name)
    df_mes.index.name = "month"  # define um nome diferente para o índice
    df_mes = df_mes.reset_index()
    df_mes['date'] = pd.to_datetime(df_mes['month'], format='%b/%Y')
    df_mes.sort_values('date', inplace=True)

    # Para os dados anuais:
    df_ano = serie_ano.value_counts().rename(serie.name)
    df_ano.index.name = "year"  # define um nome diferente para o índice
    df_ano = df_ano.reset_index()
    df_ano['date'] = pd.to_datetime(df_ano['year'], format='%Y')
    df_ano.sort_values('date', inplace=True)

    # Plots
    fig, ax = plt.subplots(3, figsize=(15, 8))

    sns.lineplot(x='date', y=serie.name, color=c, data=df_dia, ax=ax[0])

    sns.barplot(x='month', y=serie.name, color=c, data=df_mes, ax=ax[1])
    for x, y in enumerate(df_mes[serie.name]):
        ax[1].annotate(y, xy=(x, y), ha='center', va='bottom')

    sns.barplot(x='year', y=serie.name, color=c, data=df_ano, ax=ax[2])
    for x, y in enumerate(df_ano[serie.name]):
        ax[2].annotate(y, xy=(x, y), ha='center', va='bottom')

    # Setup dos plots
    ax[0].set_title('Distribuição de {} por dia'.format(serie.name))
    ax[1].set_title('Distribuição de {} por mes'.format(serie.name))
    ax[2].set_title('Distribuição de {} por ano'.format(serie.name))

    ax[0].set_xlabel('Dias')
    ax[1].set_xlabel('Meses')
    ax[2].set_xlabel('Anos')

    ax[0].set_ylabel('Frequência')
    ax[1].set_ylabel('Frequência')
    ax[2].set_ylabel('Frequência')

    ax[0].set_ylim(0, df_dia[serie.name].max() * 1.2)
    ax[1].set_ylim(0, df_mes[serie.name].max() * 1.2)
    ax[2].set_ylim(0, df_ano[serie.name].max() * 1.2)

    plt.tight_layout()
    plt.show()

    if stats:
        # Estacionaridade: Teste Dickey-Fuller
        p_value = sm.tsa.stattools.adfuller(df_dia[serie.name])[1]
        print('\n\n' + 27 * '##' + ' Estatísticas ' + 27 * '##' + '\n')

        # Plots estatísticos
        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        sns.lineplot(x='date', y=serie.name, color=c, data=df_dia, label='Dados', ax=ax1)
        sns.lineplot(x='date', y='media movel (30)', color='firebrick',
                     data=df_dia, label='Média movel (30 dias)', ax=ax1)

        plot_acf(df_dia[serie.name], ax=ax2)
        plot_pacf(df_dia[serie.name], ax=ax3)

        ax1.set_title('Análise da série temporal {}\n Dickey-fuler: p={:.5f}'.format(serie.name, p_value))

        plt.tight_layout()
        plt.show()

    return


