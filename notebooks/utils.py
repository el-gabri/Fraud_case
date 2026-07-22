from math import floor  # Usado em plot_correlations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
# Plotly para gráficos interativos
from plotly.subplots import make_subplots
from scipy.stats import skew, kurtosis  # Usado em numeric_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Imports para séries temporais
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss, adfuller, acf, pacf  # pacf e acf já importados de graphics.tsaplots


# import plotly.graph_objects as go # Opcional, make_subplots geralmente é suficiente


def plot_histograms_features(df, features_list):
    """
    Plota histogramas e boxplots para uma lista de features de um DataFrame.

    Para cada feature na lista, um histograma com KDE (Kernel Density Estimate)
    e um boxplot são gerados lado a lado. Uma linha vertical tracejada vermelha
    indica a média da feature no histograma.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    features_list (list): Lista de strings com os nomes das colunas (features) a serem plotadas.
    """
    if not isinstance(features_list, list) or not features_list:
        print("A lista de features está vazia ou não é uma lista válida.")
        return
    if df.empty:
        print("O DataFrame está vazio.")
        return

    num_features = len(features_list)
    fig, axs = plt.subplots(
        nrows=num_features, ncols=2, figsize=(12, 2.5 * num_features), layout="tight"
    )

    # Garante que axs seja sempre um array 2D, mesmo com uma única feature
    if num_features == 1:
        axs = np.array([axs])

    for idx, feature_name in enumerate(features_list):
        if feature_name not in df.columns:
            print(f"Aviso: Feature '{feature_name}' não encontrada no DataFrame. Pulando.")
            # Opcionalmente, desabilitar os eixos para a feature ausente
            if num_features > 0:  # Evita erro se axs não for array
                axs[idx, 0].set_visible(False)
                axs[idx, 1].set_visible(False)
            continue

        # Histograma
        sns.histplot(data=df, x=feature_name, kde=True, ax=axs[idx, 0], color="navy")
        axs[idx, 0].set_title(f"Histograma de {feature_name}")
        axs[idx, 0].axvline(x=df[feature_name].mean(), color="red", linestyle="--",
                            label=f"Média: {df[feature_name].mean():.2f}")
        axs[idx, 0].legend(fontsize='small')
        axs[idx, 0].set_xlabel(feature_name)
        axs[idx, 0].set_ylabel("Frequência")

        # Boxplot
        sns.boxplot(data=df, x=feature_name, ax=axs[idx, 1], color="crimson")
        axs[idx, 1].set_title(f"Boxplot de {feature_name}")
        axs[idx, 1].set_xlabel(feature_name)

    plt.suptitle("Distribuição das Features Selecionadas", fontsize=16, y=1.02)  # Título geral
    plt.show()


def plot_corr_heatmap(df, figsize=(15, 10), method='pearson'):
    """
    Plota um heatmap da matriz de correlação para as colunas numéricas do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    figsize (tuple, opcional): Tamanho da figura para o plot. Padrão é (15, 10).
    method (str, opcional): Método de correlação ('pearson', 'kendall', 'spearman').
                            Padrão é 'pearson'.
    """
    if df.empty:
        print("O DataFrame está vazio.")
        return

    # Seleciona apenas colunas numéricas para o cálculo da correlação
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        print("Nenhuma coluna numérica encontrada no DataFrame para calcular a correlação.")
        return

    corr_matrix = numeric_df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara para o triângulo superior

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap="OrRd",  # Paleta de cores (Laranja-Vermelho)
        annot=True,  # Mostrar os valores de correlação no heatmap
        fmt=".2f",  # Formato dos números (duas casas decimais)
        linewidths=0.33,
        annot_kws={"fontsize": "x-small"},  # Tamanho da fonte das anotações
        mask=mask,  # Aplicar a máscara
        cbar_kws={"shrink": .8}  # Ajusta o tamanho da barra de cores
    )
    plt.title(f"Matriz de Correlação ({method.title()})", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, figsize=(12, 8)):
    """
    Plota a importância das features de um modelo treinado.

    Assume que o modelo possui o atributo `feature_importances_`.

    Parâmetros:
    model: Modelo treinado (ex: RandomForest, GradientBoosting de scikit-learn).
           Deve possuir o atributo `feature_importances_`.
    feature_names (list): Lista de nomes das features, na mesma ordem que foram
                          usadas para treinar o modelo.
    figsize (tuple, opcional): Tamanho da figura. Padrão é (12, 8).
    """
    if not hasattr(model, 'feature_importances_'):
        print("Erro: O modelo fornecido não possui o atributo 'feature_importances_'.")
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        print("Erro: O número de importâncias não corresponde ao número de nomes de features.")
        return

    # Calcula a importância como porcentagem
    importances_percentage = (importances / importances.sum()) * 100

    # Cria um DataFrame para facilitar a ordenação e o plot
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances_percentage
    }).sort_values(by='importance', ascending=True)

    # Plotando o gráfico
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(importance_df['feature'], importance_df['importance'], color='skyblue')

    ax.set_title("Importância das Features (Porcentagem)", fontsize=16)
    ax.set_xlabel("Importância Relativa (%)", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)

    # Adiciona os valores percentuais no final de cada barra
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.5  # Pequeno deslocamento da barra
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%",
                va='center', ha='left', fontsize='small')

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    plt.tight_layout()
    plt.show()


def calculate_confidence_interval(data_sample, confidence_level=0.95):
    """
    Calcula o intervalo de confiança para a média de uma amostra de dados.

    Assume que a amostra é grande o suficiente para usar a distribuição Normal (Z-score)
    ou que os dados são normalmente distribuídos.

    Parâmetros:
    data_sample (array-like): Amostra de dados (lista, numpy array, pd.Series).
    confidence_level (float): Nível de confiança (ex.: 0.95 para 95%). Padrão é 0.95.

    Retorna:
    tuple: Tupla contendo o limite inferior e o limite superior do intervalo de confiança.
           (lower_bound, upper_bound). Retorna (np.nan, np.nan) se os dados forem insuficientes.
    """
    data_array = np.array(data_sample)
    n = len(data_array)

    if n < 2:  # Necessário pelo menos 2 pontos para calcular o desvio padrão
        print("Aviso: Dados insuficientes para calcular o intervalo de confiança (n < 2).")
        return np.nan, np.nan

    mean = np.mean(data_array)
    std_dev = np.std(data_array, ddof=1)  # ddof=1 para desvio padrão amostral

    if std_dev == 0:  # Evita divisão por zero se todos os valores forem iguais
        print("Aviso: Desvio padrão é zero. O intervalo de confiança será igual à média.")
        return mean, mean

    # Calcula o valor crítico Z para o nível de confiança
    z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Calcula a margem de erro
    margin_of_error = z_critical * (std_dev / np.sqrt(n))

    # Calcula o intervalo de confiança
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound


def plot_forecast_vs_actual(
        y_true,
        y_pred,
        title="Previsão vs. Valores Reais",
        filter_start_date=None,  # Espera pd.Timestamp ou string 'YYYY-MM-DD'
        xlabel="Data",
        ylabel="Valor",
        legend_labels=("Real", "Previsão"),
):
    """
    Plota a série temporal de previsões junto com a série original de valores reais.

    Parâmetros:
    y_true (pd.Series): Série temporal dos valores reais (observados). Deve ter um DatetimeIndex.
    y_pred (pd.Series): Série temporal dos valores previstos. Deve ter um DatetimeIndex.
    title (str, opcional): Título do gráfico.
    filter_start_date (str ou pd.Timestamp, opcional): Data de início para filtrar os dados a serem plotados.
                                                      Formato 'YYYY-MM-DD' ou objeto Timestamp.
    xlabel (str, opcional): Rótulo do eixo X.
    ylabel (str, opcional): Rótulo do eixo Y.
    legend_labels (tuple, opcional): Rótulos para as legendas de y_true e y_pred.
    """
    if not isinstance(y_true.index, pd.DatetimeIndex) or not isinstance(y_pred.index, pd.DatetimeIndex):
        print("Erro: y_true e y_pred devem ser Series com DatetimeIndex.")
        return

    # Faz cópias para evitar modificar os originais
    y_true_plot = y_true.copy()
    y_pred_plot = y_pred.copy()

    # Filtra pela data de início, se fornecida
    if filter_start_date:
        try:
            start_date = pd.to_datetime(filter_start_date).date()
            y_true_plot = y_true_plot[y_true_plot.index.date >= start_date]
            y_pred_plot = y_pred_plot[y_pred_plot.index.date >= start_date]
        except Exception as e:
            print(f"Erro ao processar filter_start_date '{filter_start_date}': {e}. Plotando todos os dados.")

    if y_true_plot.empty and y_pred_plot.empty:
        print("Nenhum dado para plotar após a filtragem pela data.")
        return

    # Plotar valores reais e previsões
    plt.figure(figsize=(12, 6))  # Ajustado para melhor visualização
    if not y_true_plot.empty:
        y_true_plot.plot(label=legend_labels[0], linewidth=1.5, color='royalblue')
    if not y_pred_plot.empty:
        y_pred_plot.plot(label=legend_labels[1], linewidth=1.5, color='darkorange', linestyle='--')

    plt.legend(loc="upper right", fontsize=12)
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def get_dataframe_info(df: pd.DataFrame):
    """
    Gera um DataFrame com informações básicas sobre as colunas de outro DataFrame.

    Inclui tipo de dados, contagem de nulos, percentual de nulos e número de valores únicos.

    Parâmetros:
    df (pd.DataFrame): DataFrame a ser analisado.

    Retorna:
    pd.DataFrame: DataFrame contendo as informações sumarizadas.
    """
    if df.empty:
        print("O DataFrame de entrada está vazio.")
        return pd.DataFrame()

    info_dict = {
        'Tipo da Coluna': df.dtypes,
        'Nº Nulos': df.isnull().sum(),
        '% Nulos': df.isnull().sum() / len(df) * 100,
        'Nº Únicos': df.nunique()
    }
    info_df = pd.DataFrame(info_dict)
    return info_df.reset_index().rename(columns={'index': 'Coluna'})


def calculate_correlations(series_x: pd.Series, series_y: pd.Series):
    """
    Calcula e imprime as correlações de Pearson, Kendall e Spearman entre duas séries.

    Parâmetros:
    series_x (pd.Series): Primeira série de valores numéricos.
    series_y (pd.Series): Segunda série de valores numéricos, de mesmo tamanho que series_x.

    Retorna:
    dict: Dicionário com os valores de cada correlação.
          Ex: {'pearson': [0.85], 'kendall': [0.70], 'spearman': [0.82]}
    """
    if len(series_x) != len(series_y):
        print("Erro: As séries devem ter o mesmo tamanho para calcular a correlação.")
        return {}
    if series_x.isnull().any() or series_y.isnull().any():
        print("Aviso: As séries contêm valores nulos. A correlação pode ser afetada ou resultar em NaN.")

    print('Correlações entre {} e {}:\n'.format(series_x.name if series_x.name else 'Série X',
                                                series_y.name if series_y.name else 'Série Y'))
    corr_dict = {}
    methods = ['pearson', 'kendall', 'spearman']
    for method in methods:
        try:
            corr_value = series_x.corr(series_y, method=method)
            corr_dict[method] = [corr_value]  # Mantém formato de lista como no original
            print(f'{method.title()}: {corr_value:.4f}')
        except Exception as e:
            print(f"Erro ao calcular correlação {method.title()}: {e}")
            corr_dict[method] = [np.nan]
    return corr_dict


def plot_rolling_correlations(series_x: pd.Series, series_y: pd.Series, series_time_index: pd.Series,
                              methods_list=None, window_size=30):
    """
    Calcula e visualiza as correlações móveis (rolling correlations) entre duas séries.

    Parâmetros:
    series_x (pd.Series): Primeira série numérica.
    series_y (pd.Series): Segunda série numérica.
    series_time_index (pd.Series): Série de datetime, usada como índice para o plot.
                                   Deve ter comprimento compatível após o cálculo da janela móvel.
    methods_list (list, opcional): Lista de métodos de correlação a serem aplicados.
                                   O primeiro método da lista é usado para anotar min/max no gráfico.
                                   Padrão: ['pearson', 'kendall', 'spearman'].
    window_size (int, opcional): Janela para o cálculo da correlação móvel. Padrão: 30.

    Retorna:
    pd.DataFrame: DataFrame com os valores das correlações móveis.
    """
    print(f'\nCorrelação em Janelas Móveis (Janela = {window_size}):\n')

    if methods_list is None:
        methods_list = ['pearson', 'kendall', 'spearman']

    rolling_corr_df = pd.DataFrame()
    for method in methods_list:
        rolling_corr_df[method] = series_x.rolling(window=window_size).corr(series_y, method=method)

    # Remove NaNs iniciais e ajusta o índice de tempo
    # O número de NaNs no início será window_size - 1
    rolling_corr_df = rolling_corr_df.iloc[window_size - 1:].reset_index(drop=True)

    if len(rolling_corr_df) != len(series_time_index.iloc[window_size - 1:]):
        print("Aviso: Incompatibilidade de tamanho entre correlações móveis e índice de tempo fornecido.")
        # Tenta usar o índice original da series_x (ou series_y) se for DatetimeIndex
        if isinstance(series_x.index, pd.DatetimeIndex):
            rolling_corr_df.index = series_x.index[window_size - 1: window_size - 1 + len(rolling_corr_df)]
        else:  # Fallback para um RangeIndex
            print("Usando RangeIndex para o plot de correlação móvel.")
    else:
        rolling_corr_df.index = series_time_index.iloc[window_size - 1: window_size - 1 + len(rolling_corr_df)]

    rolling_corr_df.reset_index(inplace=True)  # 'index' agora é a coluna de data/hora

    # Visualização
    plt.figure(figsize=(14, 7))  # Ajustado para melhor visualização
    time_col_name = rolling_corr_df.columns[0]  # Nome da coluna de índice (data/hora)

    for method in methods_list:
        if method in rolling_corr_df.columns:
            sns.lineplot(x=time_col_name, y=method, data=rolling_corr_df, label=method.title())

    # Anotações para o primeiro método da lista
    if methods_list[0] in rolling_corr_df.columns and not rolling_corr_df[methods_list[0]].empty:
        first_method_series = rolling_corr_df[methods_list[0]]
        min_val = first_method_series.min()
        min_time = rolling_corr_df.loc[first_method_series.idxmin(), time_col_name]
        plt.annotate(f'Min ({methods_list[0].title()}): {min_val:.2f}', xy=(min_time, min_val),
                     xytext=(min_time, min_val - 0.15 if min_val > -0.85 else min_val + 0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     ha='center', va='center', fontsize='small')

        max_val = first_method_series.max()
        max_time = rolling_corr_df.loc[first_method_series.idxmax(), time_col_name]
        plt.annotate(f'Max ({methods_list[0].title()}): {max_val:.2f}', xy=(max_time, max_val),
                     xytext=(max_time, max_val + 0.15 if max_val < 0.85 else max_val - 0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     ha='center', va='center', fontsize='small')

    plt.xlabel(f'Janela de {window_size} Períodos ({time_col_name})')
    plt.ylabel('Valor da Correlação')
    plt.title(f'Correlações Móveis entre {series_x.name or "Série X"} e {series_y.name or "Série Y"}', fontsize=16)
    plt.ylim(-1.1, 1.1)
    plt.legend(title="Métodos")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return rolling_corr_df.set_index(time_col_name)[methods_list]  # Retorna com índice de tempo


def plot_shifted_correlation(series_x: pd.Series, series_y: pd.Series, method='pearson', max_shift=30,
                             limit_y_axis=True):
    """
    Calcula e visualiza a correlação entre uma série e outra série deslocada no tempo.

    Parâmetros:
    series_x (pd.Series): Primeira série numérica (referência).
    series_y (pd.Series): Segunda série numérica (a ser deslocada).
    method (str, opcional): Método de correlação. Padrão: 'pearson'.
    max_shift (int, opcional): Número máximo de períodos para deslocar series_y (para trás). Padrão: 30.
    limit_y_axis (bool, opcional): Se True, limita o eixo Y do gráfico de correlação entre -1 e 1. Padrão: True.

    Retorna:
    pd.DataFrame: DataFrame contendo os períodos de deslocamento e os valores de correlação.
    """
    print(f'\nCorrelação com Deslocamento Temporal (Máx. Deslocamento = {max_shift}):\n')

    # Range de deslocamento (negativo significa que series_y está "atrasada" em relação a series_x)
    shift_range = range(-max_shift, 1)  # Inclui deslocamento 0

    # Calcula a correlação para cada deslocamento
    correlations = [series_x.corr(series_y.shift(s), method=method) for s in shift_range]

    # Cria DataFrame com resultados
    corr_df = pd.DataFrame({'deslocamento': shift_range, method: correlations})

    # Visualização
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=False)  # sharex=False pois eixos X são diferentes

    # Plot das séries originais normalizadas para comparação visual
    sns.lineplot(x=series_x.index, y=(series_x - series_x.mean()) / series_x.std(),
                 label=f'{series_x.name or "Série X"} (Normalizada)', ax=ax[0], color='blue')
    sns.lineplot(x=series_y.index, y=(series_y - series_y.mean()) / series_y.std(),
                 label=f'{series_y.name or "Série Y"} (Normalizada)', ax=ax[0], color='red', linestyle='--')
    ax[0].set_title('Séries Normalizadas para Comparação Visual')
    ax[0].set_xlabel('Tempo')
    ax[0].set_ylabel('Valor Normalizado')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)

    # Plot da correlação em função do deslocamento
    sns.lineplot(x='deslocamento', y=method, data=corr_df, ax=ax[1], marker='o', color='green')

    # Demarcação da correlação máxima (em módulo)
    if not corr_df[method].empty:
        abs_corr = corr_df[method].abs()
        if not abs_corr.empty:
            max_abs_corr_idx = abs_corr.idxmax()
            max_corr_value = corr_df.loc[max_abs_corr_idx, method]
            optimal_shift = corr_df.loc[max_abs_corr_idx, 'deslocamento']

            ax[1].axvline(optimal_shift, color='purple', linestyle=':', linewidth=2,
                          label=f'Desloc. Ótimo: {optimal_shift}')
            ax[1].annotate(f'Deslocamento: {optimal_shift}\nCorr: {max_corr_value:.2f}',
                           xy=(optimal_shift, max_corr_value),
                           xytext=(optimal_shift + 0.05 * max_shift,
                                   max_corr_value * 0.9 if max_corr_value > 0 else max_corr_value * 1.1),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                           ha='left', va='center', fontsize='small', color='purple')

    ax[1].set_title(f'Correlação ({method.title()}) vs. Deslocamento de "{series_y.name or "Série Y"}"')
    ax[1].set_xlabel(f'Deslocamento de "{series_y.name or "Série Y"}" (Períodos)')
    ax[1].set_ylabel(f'Correlação ({method.title()})')
    ax[1].axhline(0, color='black', linestyle='--', linewidth=0.7)  # Linha de correlação zero
    if limit_y_axis:
        ax[1].set_ylim(-1.05, 1.05)
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].legend(fontsize='small')

    plt.tight_layout()
    plt.show()

    return corr_df


def calculate_moving_average(series: pd.Series, window_n: int):
    """
    Calcula a média dos últimos `window_n` observações de uma série.

    Parâmetros:
    series (pd.Series): Série temporal.
    window_n (int): Número de observações para incluir na média móvel (janela).

    Retorna:
    float: A média móvel das últimas `window_n` observações.
           Retorna np.nan se a série tiver menos de `window_n` pontos.
    """
    if len(series) < window_n:
        # print(f"Aviso: A série tem menos de {window_n} observações. Não é possível calcular a média móvel completa.")
        return np.nan  # Ou poderia retornar a média dos pontos disponíveis, se desejado.
    return np.mean(series.iloc[-window_n:])  # .mean() é mais direto que np.average para Series


def normalize_min_max(series: pd.Series):
    """
    Normaliza uma série de dados usando a escala Min-Max (valores entre 0 e 1).

    Parâmetros:
    series (pd.Series): Série a ser normalizada.

    Retorna:
    pd.Series: Série normalizada.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:  # Evita divisão por zero se todos os valores forem iguais
        return pd.Series(np.zeros_like(series), index=series.index, name=series.name)
    return (series - min_val) / (max_val - min_val)


def mean_absolute_error_custom(y_true, y_predictions):  # Renomeado para evitar conflito com sklearn
    """
    Calcula o Erro Absoluto Médio (MAE).

    Parâmetros:
    y_true (array-like): Valores verdadeiros (observados).
    y_predictions (array-like): Valores previstos pelo modelo.

    Retorna:
    float: Valor do MAE.
    """
    y_true_arr, y_predictions_arr = np.array(y_true), np.array(y_predictions)
    return np.mean(np.abs(y_true_arr - y_predictions_arr))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calcula o Erro Percentual Absoluto Médio (MAPE).

    Parâmetros:
    y_true (array-like): Valores verdadeiros (observados). Não deve conter zeros.
    y_pred (array-like): Valores previstos pelo modelo.

    Retorna:
    float: Valor do MAPE em porcentagem.
           Retorna np.inf ou np.nan se y_true contiver zeros.
    """
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)

    # Verifica se há zeros em y_true para evitar divisão por zero
    if np.any(y_true_arr == 0):
        print("Aviso: y_true contém zeros. MAPE pode ser indefinido ou infinito.")
        # Opção: remover zeros ou retornar um valor específico. Aqui, permite o cálculo.

    return np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100


def plot_moving_average_with_intervals(series: pd.Series, window: int, plot_intervals=True, scale=1.96,
                                       plot_anomalies=False):
    """
    Plota a média móvel de uma série temporal, com opcionais intervalos de confiança e detecção de anomalias.

    Parâmetros:
    series (pd.Series): Série temporal com DatetimeIndex.
    window (int): Tamanho da janela da média móvel (número de observações).
    plot_intervals (bool, opcional): Se True, plota os intervalos de confiança. Padrão: True.
    scale (float, opcional): Fator de escala para os intervalos de confiança (multiplicador do desvio padrão).
                             Padrão: 1.96 (aproximadamente 95% de confiança para dados normais).
    plot_anomalies (bool, opcional): Se True e plot_intervals=True, destaca anomalias fora dos intervalos.
                                     Padrão: False.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        print("Aviso: O índice da série não é DatetimeIndex. O plot pode não ser ideal.")
    if series.empty:
        print("A série está vazia.")
        return

    rolling_mean = series.rolling(window=window, center=True,
                                  min_periods=1).mean()  # center=True para média móvel centrada

    plt.figure(figsize=(15, 7))  # Ajustado
    plt.title(f"Média Móvel (Janela = {window} Períodos)", fontsize=16)

    plt.plot(series.index, series, label="Valores Observados", color='lightgray', alpha=0.8)
    plt.plot(rolling_mean.index, rolling_mean, "g-", label="Média Móvel", linewidth=2)  # Linha sólida verde

    if plot_intervals:
        # Usa a série original para calcular o erro em relação à média móvel
        # Considera apenas o período onde a média móvel é mais estável (após a janela inicial)
        valid_period_series = series.iloc[window // 2: len(series) - window // 2]  # Ajuste para média centrada
        valid_rolling_mean = rolling_mean.iloc[window // 2: len(series) - window // 2]

        if not valid_period_series.empty and not valid_rolling_mean.empty:
            errors = valid_period_series - valid_rolling_mean
            mae = mean_absolute_error_custom(valid_period_series, valid_rolling_mean)  # Usa a função customizada
            deviation = np.std(errors)

            lower_bound = rolling_mean - (mae + scale * deviation)
            upper_bound = rolling_mean + (mae + scale * deviation)

            plt.plot(upper_bound.index, upper_bound, "r--", label="Intervalo de Confiança (Limite Superior/Inferior)",
                     alpha=0.7)
            plt.plot(lower_bound.index, lower_bound, "r--", alpha=0.7)
            plt.fill_between(rolling_mean.index, lower_bound, upper_bound, color='red', alpha=0.1,
                             label='Área de Confiança')

            if plot_anomalies:
                anomalies = series[(series < lower_bound) | (series > upper_bound)]
                if not anomalies.empty:
                    plt.plot(anomalies.index, anomalies, "ro", markersize=8, label="Anomalias Detectadas")
                else:
                    print("Nenhuma anomalia detectada fora dos intervalos de confiança.")
        else:
            print("Não há dados suficientes no período válido para calcular intervalos de confiança.")

    plt.legend(loc="upper left")
    plt.xlabel("Data")
    plt.ylabel(series.name or "Valor")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def remove_collinear_features(df_features, threshold=0.9):
    """
    Remove features colineares de um DataFrame.

    Iterativamente remove uma de cada par de features cuja correlação (em valor absoluto)
    excede o `threshold` especificado.

    Parâmetros:
    df_features (pd.DataFrame): DataFrame contendo apenas as features (colunas numéricas).
    threshold (float, opcional): Limiar de correlação para considerar features como colineares.
                                 Padrão é 0.9.

    Retorna:
    pd.DataFrame: DataFrame com as features colineares removidas.
    """
    x = df_features.copy()  # Trabalha com uma cópia
    corr_matrix = x.corr().abs()  # Usa valor absoluto da correlação
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Triângulo superior sem a diagonal

    cols_to_drop = set()
    for column in upper_triangle.columns:
        if column in cols_to_drop:  # Se a coluna já foi marcada para remoção, pula
            continue
        highly_correlated_with_column = upper_triangle[upper_triangle[column] >= threshold].index
        for correlated_feature in highly_correlated_with_column:
            if correlated_feature not in cols_to_drop:  # Evita remover ambas as features de um par se uma já foi marcada
                # Decide qual remover: pode ser a primeira encontrada, ou baseada em algum critério (ex: variância)
                # Aqui, simplesmente adiciona a 'correlated_feature' para remoção.
                # Poderia ser mais sofisticado, por exemplo, mantendo a feature com maior variância.
                print(
                    f"Feature '{correlated_feature}' é altamente correlacionada com '{column}' (corr: {corr_matrix.loc[correlated_feature, column]:.2f}). Removendo '{correlated_feature}'.")
                cols_to_drop.add(correlated_feature)

    df_filtered = x.drop(columns=list(cols_to_drop))
    print(f"\nFeatures removidas por alta colinearidade (threshold > {threshold}): {list(cols_to_drop)}")
    print(
        f"Número de features originais: {len(x.columns)}, Número de features após remoção: {len(df_filtered.columns)}")
    return df_filtered


def calculate_weighted_average(series_data: pd.Series, weights_list: list):
    """
    Calcula a média ponderada de uma série.

    Assume que a lista de pesos `weights_list` está ordenada de forma que o primeiro peso
    se aplica à observação mais recente da série (último elemento), o segundo peso à
    penúltima observação, e assim por diante. O número de pesos deve ser menor ou igual
    ao número de observações na `series_data` que serão consideradas.

    Parâmetros:
    series_data (pd.Series): Série de dados.
    weights_list (list): Lista de pesos. A soma dos pesos não precisa ser 1 (será normalizada).

    Retorna:
    float: A média ponderada. Retorna np.nan se houver problemas com os inputs.
    """
    if len(weights_list) == 0:
        print("Erro: Lista de pesos está vazia.")
        return np.nan
    if len(series_data) < len(weights_list):
        print(
            f"Aviso: A série tem {len(series_data)} observações, menos que o número de pesos ({len(weights_list)}). Usando apenas as últimas {len(series_data)} observações.")
        relevant_series_data = series_data.iloc[-len(series_data):]
        relevant_weights = np.array(weights_list[:len(series_data)])
    else:
        relevant_series_data = series_data.iloc[
                               -len(weights_list):]  # Pega as últimas N observações, onde N é o número de pesos
        relevant_weights = np.array(weights_list)

    # Normaliza os pesos para que somem 1, se já não somarem
    # relevant_weights = relevant_weights / np.sum(relevant_weights) # Opcional, np.average faz isso.

    if np.sum(relevant_weights) == 0:  # Evita divisão por zero se todos os pesos forem zero
        print("Aviso: A soma dos pesos é zero. Retornando média simples ou NaN.")
        if not relevant_series_data.empty:
            return np.mean(relevant_series_data)
        return np.nan

    # np.average já lida com a normalização dos pesos se eles não somarem 1.
    return np.average(relevant_series_data, weights=relevant_weights)


def plot_compare_two_timeseries(ts1: pd.Series, ts2: pd.Series, colors=('royalblue', 'darkorange'),
                                line_styles=('-', '--'), ax=None):
    """
    Compara duas séries temporais em um mesmo gráfico, com eixos Y separados (twin axes)
    e plota suas linhas de tendência linear.

    Parâmetros:
    ts1 (pd.Series): Primeira série temporal.
    ts2 (pd.Series): Segunda série temporal.
    colors (tuple, opcional): Tupla com as cores para ts1 e ts2.
    line_styles (tuple, opcional): Tupla com os estilos de linha para as tendências de ts1 e ts2.
    ax (matplotlib.axes.Axes, opcional): Eixo Matplotlib para plotar. Se None, um novo é criado.
    """
    ts1_clean = ts1.dropna()
    ts2_clean = ts2.dropna()

    if ts1_clean.empty and ts2_clean.empty:
        print("Ambas as séries estão vazias após remover NaNs. Nada para plotar.")
        return

    create_new_fig = ax is None
    if create_new_fig:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    else:
        ax1 = ax

    # Plot da primeira série (ts1)
    if not ts1_clean.empty:
        ax1.plot(ts1_clean.index, ts1_clean, color=colors[0], label=ts1_clean.name or 'Série 1')
        # Linha de tendência para ts1
        x_numeric_ts1 = np.arange(len(ts1_clean.index))  # Converte índice de data para numérico para polyfit
        if len(x_numeric_ts1) > 1:  # Polyfit precisa de pelo menos 2 pontos
            trend_coeffs_ts1 = np.polyfit(x_numeric_ts1, ts1_clean, 1)
            trend_line_ts1 = np.poly1d(trend_coeffs_ts1)
            ax1.plot(ts1_clean.index, trend_line_ts1(x_numeric_ts1), linestyle=line_styles[0], color=colors[0],
                     alpha=0.7, label=f'Tendência {ts1_clean.name or "Série 1"}')
    ax1.set_xlabel("Data")
    ax1.set_ylabel(ts1_clean.name or 'Série 1', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.grid(None)  # Remove grid padrão do eixo primário se houver

    # Cria um segundo eixo Y para a segunda série (ts2)
    ax2 = ax1.twinx()
    if not ts2_clean.empty:
        ax2.plot(ts2_clean.index, ts2_clean, color=colors[1], label=ts2_clean.name or 'Série 2')
        # Linha de tendência para ts2
        x_numeric_ts2 = np.arange(len(ts2_clean.index))
        if len(x_numeric_ts2) > 1:
            trend_coeffs_ts2 = np.polyfit(x_numeric_ts2, ts2_clean, 1)
            trend_line_ts2 = np.poly1d(trend_coeffs_ts2)
            ax2.plot(ts2_clean.index, trend_line_ts2(x_numeric_ts2), linestyle=line_styles[1], color=colors[1],
                     alpha=0.7, label=f'Tendência {ts2_clean.name or "Série 2"}')
    ax2.set_ylabel(ts2_clean.name or 'Série 2', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax2.grid(None)  # Remove grid padrão do eixo secundário

    # Legenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    if create_new_fig:
        plt.title(f"Comparação entre {ts1_clean.name or 'Série 1'} e {ts2_clean.name or 'Série 2'}", fontsize=16)
        plt.tight_layout()
        plt.show()


def plot_multiple_timeseries_comparisons(df, main_var_name, comparison_vars_list, num_rows, num_cols, figsize=(15, 10)):
    """
    Plota múltiplas comparações de séries temporais em uma grade de subplots.

    Cada subplot compara `main_var_name` com uma das variáveis em `comparison_vars_list`.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo todas as séries temporais.
    main_var_name (str): Nome da variável principal a ser comparada.
    comparison_vars_list (list): Lista de nomes de variáveis para comparar com `main_var_name`.
    num_rows (int): Número de linhas na grade de subplots.
    num_cols (int): Número de colunas na grade de subplots.
    figsize (tuple, opcional): Tamanho da figura geral.
    """
    if main_var_name not in df.columns:
        print(f"Erro: Variável principal '{main_var_name}' não encontrada no DataFrame.")
        return

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True)  # sharex para alinhar eixos de tempo
    axes = axes.flatten()  # Transforma a matriz de eixos em um array 1D para fácil iteração

    vars_to_compare = [var for var in comparison_vars_list if var != main_var_name and var in df.columns]

    if not vars_to_compare:
        print("Nenhuma variável válida para comparação encontrada na lista fornecida.")
        if num_rows > 0 and num_cols > 0:  # Limpa a figura se foi criada
            fig.clear()
            plt.close(fig)
        return

    for i, var_name in enumerate(vars_to_compare):
        if i < len(axes):  # Garante que não tentemos acessar um eixo inexistente
            current_ax = axes[i]
            plot_compare_two_timeseries(df[main_var_name], df[var_name], ax=current_ax)
            current_ax.set_title(f'{main_var_name} vs. {var_name}', fontsize='medium')
        else:
            print(
                f"Aviso: Grade de subplots ({num_rows}x{num_cols}) é pequena demais para todas as comparações. '{var_name}' não será plotada.")
            break

    # Remove eixos não utilizados, se houver
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Comparações Múltiplas com {main_var_name}", fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Ajusta para o suptitle
    plt.show()


def predict_sarima_rolling_forecast(data_series, pred_col_name, cutoff_col_name, cutoff_ref_value,
                                    forecast_steps, series_order, seasonal_order, exog_data=None):
    """
    Realiza previsões SARIMA usando uma abordagem de janela deslizante ou expansiva (dependendo do uso).
    Esta função parece projetada para um tipo específico de validação cruzada ou simulação de previsão.

    Parâmetros:
    data_series (pd.DataFrame): DataFrame contendo a série a ser prevista e a coluna de corte.
    pred_col_name (str): Nome da coluna contendo os valores a serem previstos (endógena).
    cutoff_col_name (str): Nome da coluna usada para definir os pontos de corte do treinamento.
    cutoff_ref_value: Valor de referência na `cutoff_col_name` para iniciar os cortes.
                      As previsões são feitas para cada valor único em `cutoff_col_name` >= `cutoff_ref_value`.
    forecast_steps (int): Número de passos à frente para prever.
    series_order (tuple): Ordem (p,d,q) do componente não sazonal do SARIMA.
    seasonal_order (tuple): Ordem (P,D,Q,s) do componente sazonal do SARIMA.
    exog_data (pd.DataFrame ou pd.Series, opcional): Dados exógenos para o modelo SARIMAX.
                                                    Deve estar alinhado com `data_series`.

    Retorna:
    pd.DataFrame: DataFrame contendo as previsões. Cada linha corresponde a um ponto de corte,
                  e as colunas são as previsões para os `forecast_steps` futuros.
    """
    # Filtra os valores únicos da coluna de corte a partir do valor de referência
    slice_values = sorted(data_series.loc[data_series[cutoff_col_name] >= cutoff_ref_value, cutoff_col_name].unique())

    results_df = pd.DataFrame()

    if not slice_values:
        print(f"Nenhum valor de corte encontrado para '{cutoff_col_name}' >= {cutoff_ref_value}.")
        return results_df

    for i, current_cutoff_value in enumerate(slice_values):
        # A lógica original `if slice_value == slice_values[-1]: continue`
        # faria com que o último período de corte não fosse usado para gerar uma previsão.
        # Se a intenção é prever após o último dado de treino disponível, essa condição pode ser removida
        # ou ajustada dependendo do cenário de uso (ex: se o último corte é para teste final).
        # Para uma simulação de rolling forecast, normalmente se prevê após cada período de treino.

        # Dados de treinamento são todos até o `current_cutoff_value` (inclusive)
        train_df = data_series[data_series[cutoff_col_name] <= current_cutoff_value].copy()

        if train_df.empty:
            print(f"DataFrame de treino vazio para o corte {current_cutoff_value}. Pulando.")
            continue

        train_endog = train_df[pred_col_name]
        train_exog = None
        if exog_data is not None:
            # Garante que os dados exógenos de treino correspondam ao período de treino endógeno
            train_exog = exog_data.loc[train_endog.index]
            if len(train_exog) != len(train_endog):
                print(
                    f"Aviso: Incompatibilidade de tamanho entre dados endógenos e exógenos para o corte {current_cutoff_value}.")
                # Poderia tentar reindexar ou tratar de outra forma. Por ora, prossegue.

        try:
            model = SARIMAX(endog=train_endog,
                            exog=train_exog,  # Adiciona exog se fornecido
                            order=series_order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,  # Comum para SARIMA
                            enforce_invertibility=False)  # Comum para SARIMA

            # O método 'powell' pode ser lento. 'lbfgs' é frequentemente mais rápido.
            model_fit_results = model.fit(disp=False)  # disp=False para suprimir output de convergência

            # Ponto de início da previsão é o final do conjunto de treino
            start_pred_idx = len(train_endog)
            # Ponto final da previsão
            end_pred_idx = start_pred_idx + forecast_steps - 1

            # Previsão para os próximos `forecast_steps`
            # Precisa de exógenos futuros se o modelo foi treinado com exógenos
            future_exog = None
            if exog_data is not None:
                # Tenta obter os exógenos para o período de previsão
                # Isso assume que `exog_data` contém os valores futuros necessários
                # e que seu índice está alinhado ou pode ser alinhado.
                # Esta parte é crucial e pode ser complexa dependendo de como `exog_data` é estruturado.
                # Simplificação: assume que `exog_data` cobre o período de previsão.
                # Uma abordagem mais robusta seria passar `exog_data` completo e deixar `predict` fatiar.
                # Ou, se `exog_data` só vai até o fim do treino, então só se pode prever 1 passo se
                # os exógenos futuros não forem conhecidos/previstos separadamente.
                # Para previsão multi-passo com exógenos, os valores futuros dos exógenos DEVEM ser fornecidos.

                # Se exog_data está alinhado com data_series original:
                original_data_index = data_series.index
                last_train_date = train_endog.index[-1]

                # Encontra o índice no `data_series` original que corresponde ao fim do treino
                loc_end_train_in_original = original_data_index.get_loc(last_train_date)

                start_exog_pred_loc = loc_end_train_in_original + 1
                end_exog_pred_loc = loc_end_train_in_original + forecast_steps

                if end_exog_pred_loc < len(exog_data):
                    future_exog = exog_data.iloc[start_exog_pred_loc: end_exog_pred_loc + 1]
                    if len(future_exog) != forecast_steps:
                        print(
                            f"Aviso: Número de exógenos futuros ({len(future_exog)}) não corresponde a forecast_steps ({forecast_steps}) para corte {current_cutoff_value}.")
                        future_exog = None  # Anula se não bater
                else:
                    print(
                        f"Aviso: Dados exógenos futuros insuficientes para corte {current_cutoff_value}. Previsão pode falhar ou ser sem exog.")
                    future_exog = None

            predictions = model_fit_results.predict(start=start_pred_idx, end=end_pred_idx, exog=future_exog)

            # Organiza as previsões em um formato de linha para o DataFrame de resultados
            predictions_row = predictions.to_frame(name=current_cutoff_value).T
            pred_cols_names = [f'{pred_col_name}_pred_step{j + 1}' for j in range(len(predictions_row.columns))]
            predictions_row.columns = pred_cols_names
            predictions_row.index.name = cutoff_col_name  # Nomeia o índice com o nome da coluna de corte

            results_df = pd.concat([results_df, predictions_row])

        except Exception as e:
            print(f"Erro ao treinar ou prever SARIMA para o corte {current_cutoff_value}: {e}")
            # Adiciona uma linha de NaNs ou continua, dependendo da robustez desejada
            error_row = pd.DataFrame([[np.nan] * forecast_steps],
                                     columns=[f'{pred_col_name}_pred_step{j + 1}' for j in range(forecast_steps)],
                                     index=[current_cutoff_value])
            error_row.index.name = cutoff_col_name
            results_df = pd.concat([results_df, error_row])

    return results_df


def plot_time_series_decomposition_plotly(decomposition_results, series_col_name=None,
                                          shared_x=True, shared_y=False,
                                          title_suffix=""):
    """
    Plota os componentes de uma decomposição de série temporal (observado, tendência, sazonalidade, resíduos)
    usando Plotly para interatividade.

    Compatível com resultados de `statsmodels.tsa.seasonal.seasonal_decompose` (que tem `dec.observed`, etc.)
    e `statsmodels.tsa.seasonal.STL` (que tem `dec.observed[col]`, etc., se multivariada no input).

    Parâmetros:
    decomposition_results: Objeto resultado da decomposição (ex: de `seasonal_decompose` ou `STL`).
    series_col_name (str, opcional): Nome da coluna da série observada, necessário se `decomposition_results`
                                     for de STL e a entrada original para STL foi um DataFrame com múltiplas colunas.
                                     Se a entrada para STL foi uma Série, ou se usando `seasonal_decompose`,
                                     este parâmetro pode ser omitido.
    shared_x (bool, opcional): Se os eixos X devem ser compartilhados entre os subplots. Padrão: True.
    shared_y (bool, opcional): Se os eixos Y devem ser compartilhados. Padrão: False.
    title_suffix (str, opcional): Sufixo a ser adicionado ao título principal do gráfico.

    Retorna:
    plotly.graph_objects.Figure: Figura Plotly contendo os gráficos de decomposição.
    """

    fig = make_subplots(rows=4, cols=1, shared_xaxes=shared_x, shared_yaxes=shared_y,
                        vertical_spacing=0.05,
                        subplot_titles=('Série Observada', 'Tendência (Trend)',
                                        'Sazonalidade (Seasonal)', 'Resíduos (Residual)'))

    # Determina como acessar os componentes baseado no tipo de resultado da decomposição
    # seasonal_decompose retorna Series diretamente nos atributos (dec.observed, dec.trend, etc.)
    # STL, se input foi DataFrame, retorna DataFrame e precisa de `series_col_name`

    # Observado
    if hasattr(decomposition_results, 'observed'):
        if isinstance(decomposition_results.observed, pd.Series):
            observed_series = decomposition_results.observed
        elif isinstance(decomposition_results.observed,
                        pd.DataFrame) and series_col_name in decomposition_results.observed.columns:
            observed_series = decomposition_results.observed[series_col_name]
        elif series_col_name is None and isinstance(decomposition_results.observed, pd.DataFrame) and len(
                decomposition_results.observed.columns) == 1:
            observed_series = decomposition_results.observed.iloc[:, 0]  # Pega a primeira coluna se só houver uma
            print(
                f"Aviso: 'series_col_name' não fornecido para resultado STL de DataFrame. Usando a primeira coluna: {observed_series.name}")
        else:
            raise ValueError(
                f"Não foi possível acessar a série observada. Verifique 'decomposition_results' e 'series_col_name' (se aplicável para STL com DataFrame). Colunas disponíveis: {list(decomposition_results.observed.columns) if isinstance(decomposition_results.observed, pd.DataFrame) else 'N/A'}")
    else:
        raise AttributeError("O objeto 'decomposition_results' não possui o atributo 'observed'.")

    # Tendência, Sazonalidade, Resíduos (assumindo que seguem a mesma estrutura que 'observed')
    trend_series = getattr(decomposition_results, 'trend')
    seasonal_series = getattr(decomposition_results, 'seasonal')
    resid_series = getattr(decomposition_results, 'resid')

    if isinstance(trend_series, pd.DataFrame): trend_series = trend_series[
        series_col_name if series_col_name else observed_series.name]
    if isinstance(seasonal_series, pd.DataFrame): seasonal_series = seasonal_series[
        series_col_name if series_col_name else observed_series.name]
    if isinstance(resid_series, pd.DataFrame): resid_series = resid_series[
        series_col_name if series_col_name else observed_series.name]

    fig.add_scatter(row=1, col=1, x=observed_series.index, y=observed_series, name='Observada', line_color='blue')
    fig.add_scatter(row=2, col=1, x=trend_series.index, y=trend_series, name='Tendência', line_color='green')
    fig.add_scatter(row=3, col=1, x=seasonal_series.index, y=seasonal_series, name='Sazonalidade', line_color='orange')
    fig.add_scatter(row=4, col=1, x=resid_series.index, y=resid_series, name='Resíduos', line_color='purple',
                    mode='markers', marker=dict(size=3))

    main_title = f'Decomposição da Série Temporal: {observed_series.name or "Série"}'
    if title_suffix:
        main_title += f" ({title_suffix})"

    fig.update_layout(
        title=main_title,
        title_x=0.5,  # Centraliza o título
        showlegend=False,  # Legendas já estão nos títulos dos subplots ou são auto-explicativas
        height=800,
        # width=1000, # Comentado para permitir ajuste automático ou ser definido externamente
        plot_bgcolor='rgba(245,245,245,1)',  # Cor de fundo suave
        paper_bgcolor='rgba(255,255,255,1)',  # Cor do papel
        xaxis_type="date",
        xaxis_rangeslider_visible=False,  # Desabilita o rangeslider no eixo X principal
        xaxis1=dict(  # Configurações para o primeiro eixo X (Observado)
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1a", step="year", stepmode="backward"),  # "1y" para "1 ano"
                    dict(count=2, label="2a", step="year", stepmode="backward"),
                    dict(count=1, label="Desde Início do Ano", step="year", stepmode="todate"),  # "YTD"
                    dict(step="all", label="Tudo")
                ])
            ),
            type="date",
        ),
        xaxis4_rangeslider_visible=True,  # Habilita rangeslider no último subplot (Resíduos)
        font=dict(family="Arial, sans-serif", size=11)  # Fonte global
    )

    # Melhora os títulos dos eixos Y para cada subplot
    fig.update_yaxes(title_text="Valor Observado", row=1, col=1)
    fig.update_yaxes(title_text="Tendência", row=2, col=1)
    fig.update_yaxes(title_text="Sazonalidade", row=3, col=1)
    fig.update_yaxes(title_text="Resíduos", row=4, col=1)

    fig.show()  # Mostra o gráfico interativo
    return fig


def check_stationarity(df_series, series_col_name, num_diffs=None, significance_level=0.05):
    """
    Verifica a estacionariedade de uma série temporal usando os testes ADFuller e KPSS.

    Parâmetros:
    df_series (pd.DataFrame ou pd.Series): DataFrame contendo a série ou a própria Série.
    series_col_name (str): Nome da coluna da série no DataFrame. Ignorado se df_series for uma Série.
    num_diffs (int, opcional): Número de diferenciações a aplicar na série antes dos testes.
                               Se None, usa a série original. Padrão: None.
    significance_level (float, opcional): Nível de significância (alpha) para os testes. Padrão: 0.05.
    """
    if isinstance(df_series, pd.DataFrame):
        if series_col_name not in df_series.columns:
            print(f"Erro: Coluna '{series_col_name}' não encontrada no DataFrame.")
            return
        series_to_test = df_series[series_col_name].dropna()  # Remove NaNs antes de diferenciar
    elif isinstance(df_series, pd.Series):
        series_to_test = df_series.dropna()
    else:
        print("Erro: 'df_series' deve ser um DataFrame ou uma Series do Pandas.")
        return

    if series_to_test.empty:
        print(
            f"A série '{series_col_name or series_to_test.name}' está vazia após remover NaNs. Teste de estacionariedade não pode ser realizado.")
        return

    if num_diffs is not None and num_diffs > 0:
        series_to_test = series_to_test.diff(num_diffs).dropna()
        print(f"Aplicada diferenciação de ordem {num_diffs}.")
        if series_to_test.empty:
            print(f"A série ficou vazia após {num_diffs} diferenciações. Teste não pode ser realizado.")
            return

    print(
        f'\n--- Teste de Estacionaridade para "{series_col_name or series_to_test.name}" (alpha={significance_level}) ---')

    # Teste ADFuller (H0: Não-estacionária)
    try:
        adf_test_result = adfuller(series_to_test)
        p_value_adf = adf_test_result[1]
        is_stationary_adf = p_value_adf < significance_level
        print(f'Teste Augmented Dickey-Fuller (ADF):')
        print(f'  Estatística do Teste: {adf_test_result[0]:.4f}')
        print(f'  P-valor: {p_value_adf:.4f}')
        print(f'Valores críticos: {adf_test_result[4]}')
        print(f'  Resultado: {"Estacionária" if is_stationary_adf else "Não-Estacionária"} (com base no p-valor)')
    except Exception as e:
        print(f"  Erro ao executar o teste ADF: {e}")

    # Teste KPSS (H0: Estacionária em torno de uma tendência determinística)
    try:
        # 'c' para constante (estacionária em torno de uma média)
        # 'ct' para constante e tendência (estacionária em torno de uma tendência linear)
        # Usar 'c' como padrão, pode ser ajustado se a série tiver uma tendência clara.
        kpss_test_result = kpss(series_to_test, regression='c', nlags='auto')
        p_value_kpss = kpss_test_result[1]
        # Para KPSS, a hipótese nula é estacionariedade.
        # Se p-valor < alpha, rejeitamos H0 (ou seja, a série é não-estacionária).
        is_stationary_kpss = p_value_kpss >= significance_level
        print(f'\nTeste Kwiatkowski-Phillips-Schmidt-Shin (KPSS):')
        print(f'  Estatística do Teste: {kpss_test_result[0]:.4f}')
        print(f'  P-valor: {p_value_kpss:.4f} (Lags usados: {kpss_test_result[2]})')
        print(f'Valores críticos: {kpss_test_result[3]}')
        print(f'  Resultado: {"Estacionária" if is_stationary_kpss else "Não-Estacionária"} (com base no p-valor)')
        if p_value_kpss < 0.01: print("  (Aviso KPSS: p-valor é muito baixo, forte evidência contra estacionariedade)")
        if p_value_kpss > 0.1: print("  (Aviso KPSS: p-valor é alto, não se pode rejeitar estacionariedade)")

    except Exception as e:
        print(f"  Erro ao executar o teste KPSS: {e}")


def plot_acf_pacf_plotly(series_data: pd.Series, num_lags=None, title_prefix=""):
    """
    Plota as funções de Autocorrelação (ACF) e Autocorrelação Parcial (PACF)
    de uma série temporal usando Plotly para interatividade.

    Parâmetros:
    series_data (pd.Series): Série temporal.
    num_lags (int, opcional): Número de lags a serem exibidos.
                              Se None, um valor padrão é calculado (min(29, N/2 - 1)).
    title_prefix (str, opcional): Prefixo para os títulos dos gráficos.

    Retorna:
    plotly.graph_objects.Figure: Figura Plotly contendo os gráficos ACF e PACF.
    """
    series_clean = series_data.dropna()
    if series_clean.empty:
        print("Série vazia após remover NaNs. Não é possível plotar ACF/PACF.")
        return None

    if num_lags is None:
        # Define um número padrão de lags, garantindo que seja razoável
        # statsmodels usa nlags = min(10 * log10(N), N - 1) para acf e pacf por padrão se alpha é especificado.
        # Aqui, uma heurística mais simples para visualização.
        num_lags = min(40, floor(len(series_clean) / 2) - 1)

    if num_lags < 1:
        print(f"Número de lags ({num_lags}) é muito pequeno. Não é possível plotar ACF/PACF.")
        return None

    try:
        # Calcula ACF e PACF com intervalos de confiança
        # alpha=0.05 para intervalo de confiança de 95%
        acf_values, confint_acf = acf(series_clean, nlags=num_lags, alpha=0.05,
                                      fft=False)  # fft=False para evitar warning com NaNs, já dropados
        pacf_values, confint_pacf = pacf(series_clean, nlags=num_lags, alpha=0.05, method='ols')  # method='ols' é comum
    except Exception as e:
        print(f"Erro ao calcular ACF/PACF: {e}")
        return None

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True, shared_yaxes=False,
                        # Y não compartilhado, pois ACF e PACF podem ter escalas diferentes
                        vertical_spacing=0.1,  # Aumenta o espaçamento
                        subplot_titles=(f'Função de Autocorrelação (ACF) - {title_prefix}'.strip(),
                                        f'Função de Autocorrelação Parcial (PACF) - {title_prefix}'.strip()))

    lags_x = np.arange(0, num_lags + 1)  # Lags de 0 a num_lags

    # Função auxiliar para plotar ACF/PACF com Plotly
    def _plotly_corr_plot(fig_handle, row, x_values, y_values, conf_intervals, plot_title):
        # Linhas verticais para cada lag (do y=0 até o valor da correlação)
        for x_val, y_val in zip(x_values[1:], y_values[1:]):  # Começa do lag 1
            fig_handle.add_shape(type="line", x0=x_val, y0=0, x1=x_val, y1=y_val,
                                 line=dict(color="grey", width=2), row=row, col=1)

        # Marcadores para os valores de correlação
        fig_handle.add_scatter(row=row, col=1, x=x_values[1:], y=y_values[1:], mode='markers',
                               marker_color='#1f77b4', marker_size=8, name=plot_title)

        # Intervalo de confiança (área sombreada)
        # O confint retornado por acf/pacf é [:,0] para lower e [:,1] para upper, RELATIVO ao valor da correlação.
        # Para plotar, precisamos do valor absoluto do intervalo.
        # statsmodels.graphics.tsaplots faz (valor - limite_inferior) e (limite_superior - valor)
        # Aqui, vamos usar o confint diretamente se ele for absoluto, ou calcular.
        # O confint de statsmodels.tsa.stattools.acf/pacf é [valor - erro, valor + erro]
        # Então, confint[:,0] é o limite inferior e confint[:,1] é o limite superior.

        # Plotando o intervalo de confiança (começando do lag 1)
        fig_handle.add_scatter(row=row, col=1, x=x_values[1:], y=conf_intervals[1:, 1], mode='lines',  # Limite superior
                               line_color='rgba(32, 146, 230,0.3)', showlegend=False, hoverinfo='skip')
        fig_handle.add_scatter(row=row, col=1, x=x_values[1:], y=conf_intervals[1:, 0], mode='lines',  # Limite inferior
                               fillcolor='rgba(32, 146, 230,0.2)', fill='tonexty',
                               # Preenche até a linha anterior (superior)
                               line_color='rgba(32, 146, 230,0.3)', showlegend=False, hoverinfo='skip')

        # Linha horizontal em y=0
        fig_handle.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", row=row, col=1)
        fig_handle.update_yaxes(title_text="Correlação", row=row, col=1, range=[-1.05, 1.05])

    # Plot ACF
    _plotly_corr_plot(fig, 1, lags_x, acf_values, confint_acf, 'ACF')
    # Plot PACF
    _plotly_corr_plot(fig, 2, lags_x, pacf_values, confint_pacf, 'PACF')

    fig.update_xaxes(title_text="Lag", row=2, col=1)  # Adiciona título ao eixo X apenas no último plot
    fig.update_layout(height=600, showlegend=False, title_x=0.5,
                      plot_bgcolor='rgba(245,245,245,1)', paper_bgcolor='rgba(255,255,255,1)')

    fig.show()
    return fig


def summarize_dataframe_characteristics(df: pd.DataFrame):
    """
    Imprime características básicas de um DataFrame: dimensões, duplicatas e contagem de vazios por coluna.

    Parâmetros:
    df (pd.DataFrame): DataFrame a ser analisado.
    """
    if not isinstance(df, pd.DataFrame):
        print("Erro: O input fornecido não é um DataFrame do Pandas.")
        return

    print('--- Características do DataFrame ---')
    print(f'Nº de Linhas: {df.shape[0]}')
    print(f'Nº de Colunas: {df.shape[1]}')

    num_duplicates = df.duplicated().sum()
    print(f'Nº de Linhas Duplicadas: {num_duplicates} ({num_duplicates / df.shape[0] * 100:.2f}% do total)' if df.shape[
                                                                                                                   0] > 0 else 'Nº de Linhas Duplicadas: 0')

    print('\nContagem de Valores Vazios (NaN) por Coluna:')
    if df.empty:
        print("  DataFrame está vazio, sem colunas para analisar.")
        return

    nan_counts = df.isnull().sum()
    if nan_counts.sum() == 0:
        print("  Nenhuma coluna contém valores vazios.")
    else:
        for col_name in df.columns:
            num_na = nan_counts[col_name]
            if num_na > 0:  # Mostra apenas colunas com NaNs
                percentage_na = (num_na / df.shape[0]) * 100
                print(f'  Coluna "{col_name}": {num_na} vazios ({percentage_na:.2f}%)')
    print('------------------------------------')
    return  # A função original não retornava nada, apenas imprimia.


def check_string_pattern_violations(series: pd.Series, expected_length=None, check_numeric_only=False):
    """
    Verifica uma série de strings em busca de violações de padrão:
    1. Comprimento diferente do esperado (se `expected_length` for fornecido).
    2. Presença de caracteres não numéricos (se `check_numeric_only` for True).

    Parâmetros:
    series (pd.Series): Série do Pandas contendo strings a serem verificadas.
    expected_length (int, opcional): O comprimento esperado para cada string.
                                     Se None, esta checagem é pulada.
    check_numeric_only (bool, opcional): Se True, verifica se as strings contêm apenas dígitos.
                                         Se False, esta checagem é pulada.

    Retorna:
    tuple: (violations_length, violations_non_numeric)
           violations_length (pd.Series): Booleana, True onde o comprimento é inesperado.
           violations_non_numeric (pd.Series): Booleana, True onde há caracteres não numéricos (se checado).
                                               Retorna uma Série vazia se não checado.
    """
    violations_length = pd.Series([False] * len(series), index=series.index)
    violations_non_numeric = pd.Series([False] * len(series), index=series.index)

    if expected_length is not None:
        # Converte para string para aplicar len(), tratando NaNs
        violations_length = series.astype(str).apply(
            lambda x: False if pd.isna(x) or x == 'nan' else len(x) != expected_length)
        print(f"Número de strings com comprimento diferente de {expected_length}: {violations_length.sum()}")

    if check_numeric_only:
        # Verifica se há algum caractere que não seja um dígito
        violations_non_numeric = series.astype(str).apply(
            lambda x: False if pd.isna(x) or x == 'nan' else not x.isdigit())
        print(f"Número de strings contendo caracteres não numéricos: {violations_non_numeric.sum()}")

    return violations_length, violations_non_numeric


def plot_categorical_distribution(series: pd.Series, main_color='mediumseagreen', max_categories_direct=10):
    """
    Cria um gráfico de barras horizontais para a distribuição de uma série categórica.

    Agrupa categorias menos frequentes em "Outros".
    Destaca valores "Faltante" (NaNs convertidos).

    Parâmetros:
    series (pd.Series): Série categórica a ser plotada.
    main_color (str, opcional): Cor principal para as barras. Padrão: 'mediumseagreen'.
    max_categories_direct (int, opcional): Número máximo de categorias a serem mostradas diretamente.
                                           As demais serão agrupadas em "Outros". Padrão: 10.
    """
    if not isinstance(series, pd.Series):
        print("Erro: A entrada deve ser uma Series do Pandas.")
        return

    # Trata NaNs como uma categoria "Faltante"
    series_processed = series.fillna('Faltante').copy()
    series_name = series.name or "Variável Categórica"

    # Contagem de frequência
    value_counts = series_processed.value_counts().reset_index()
    value_counts.columns = ['categoria', 'frequencia']

    # Separa "Faltante" se existir
    missing_data = value_counts[value_counts['categoria'] == 'Faltante']
    value_counts = value_counts[value_counts['categoria'] != 'Faltante']

    # Agrupa em "Outros" se houver muitas categorias
    num_other_categories = 0
    if len(value_counts) > max_categories_direct:
        num_other_categories = len(value_counts) - (max_categories_direct - 1)  # -1 para dar espaço para "Outros"
        df_top = value_counts.head(max_categories_direct - 1).copy()
        sum_others = value_counts.iloc[max_categories_direct - 1:]['frequencia'].sum()
        df_others = pd.DataFrame([{'categoria': 'Outros', 'frequencia': sum_others}])
        value_counts_plot = pd.concat([df_top, df_others])
    else:
        value_counts_plot = value_counts.copy()

    # Adiciona "Faltante" de volta, se houver
    if not missing_data.empty:
        value_counts_plot = pd.concat([value_counts_plot, missing_data]).reset_index(drop=True)

    value_counts_plot = value_counts_plot.sort_values(by='frequencia', ascending=False)

    # Configuração do Plot
    palette = {catg: main_color if catg not in ['Outros', 'Faltante']
    else 'grey' if catg == 'Outros'
    else 'black' for catg in value_counts_plot['categoria']}

    plot_height = max(4, len(value_counts_plot['categoria']) * 0.5)  # Altura dinâmica
    plt.figure(figsize=(12, plot_height))  # Largura fixa, altura dinâmica

    # Plot
    bars = sns.barplot(x='frequencia', y='categoria', data=value_counts_plot, palette=palette, orient='h')

    # Adiciona anotações (frequência e percentual)
    total_sum = value_counts_plot['frequencia'].sum()
    for i, bar_val in enumerate(value_counts_plot['frequencia']):
        percentage = (bar_val / total_sum) * 100 if total_sum > 0 else 0
        # Posição das anotações
        # Se a barra for muito pequena, anota fora. Senão, dentro.
        text_x_offset = value_counts_plot['frequencia'].max() * 0.01  # Pequeno offset

        if bar_val < (0.2 * value_counts_plot['frequencia'].max()):  # Se a barra for < 20% da maior barra
            plt.text(bar_val + text_x_offset, i, f"{bar_val} ({percentage:.1f}%)",
                     va='center', ha='left', color='black', fontsize='small')
        else:
            plt.text(bar_val - text_x_offset, i, f"{bar_val}",
                     va='center', ha='right', color='white', fontsize='small', weight='bold')
            plt.text(bar_val + text_x_offset, i, f"({percentage:.1f}%)",
                     va='center', ha='left', color='black', fontsize='small')

    # Layout
    plt.xlim(0, value_counts_plot['frequencia'].max() * 1.15)  # Espaço para anotações externas
    plt.title(f'Distribuição da Coluna: {series_name}', fontsize=16)
    plt.xlabel('Frequência Absoluta', fontsize=12)
    plt.ylabel(series_name.title().replace('_', ' '), fontsize=12)
    if num_other_categories > 0:
        plt.figtext(0.5, 0.01, f"* 'Outros' agrupa {num_other_categories} categorias menores.", ha="center",
                    fontsize=10, color="dimgray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta para título e figtext
    plt.show()


def plot_time_series_overview(series: pd.Series, series_name_override=None, main_color='darkcyan',
                              show_stats_plots=True):
    """
    Cria gráficos para análise exploratória de uma série temporal.
    Inclui distribuição diária, mensal e anual, e opcionalmente, gráficos estatísticos (ACF, PACF).

    Parâmetros:
    series (pd.Series): Série temporal com índice DatetimeIndex.
    series_name_override (str, opcional): Nome a ser usado para a série nos títulos. Se None, usa series.name.
    main_color (str, opcional): Cor principal para os gráficos. Padrão: 'darkcyan'.
    show_stats_plots (bool, opcional): Se True, exibe gráficos estatísticos (ACF, PACF) e teste Dickey-Fuller.
                                     Padrão: True.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            # Tenta converter o índice para DatetimeIndex se for um formato comum
            series_datetime_idx = series.copy()
            series_datetime_idx.index = pd.to_datetime(series_datetime_idx.index)
            print("Índice convertido para DatetimeIndex.")
        except Exception as e:
            print(f"Erro: O índice da série não é DatetimeIndex e não pôde ser convertido automaticamente: {e}")
            print("Por favor, forneça uma série com DatetimeIndex.")
            return
    else:
        series_datetime_idx = series.copy()

    series_name = series_name_override if series_name_override else series_datetime_idx.name or "Série Temporal"

    # Informações básicas
    print(f"\n--- Visão Geral da Série Temporal: {series_name} ---")
    print(f'Número de observações: {len(series_datetime_idx)}')
    print(f'Valores ausentes (NaN): {series_datetime_idx.isnull().sum()}')
    if not series_datetime_idx.empty:
        print(
            f'Período da série: {series_datetime_idx.index.min().strftime("%d/%b/%Y %H:%M")} a {series_datetime_idx.index.max().strftime("%d/%b/%Y %H:%M")}')
        time_delta = series_datetime_idx.index.max() - series_datetime_idx.index.min()
        years = time_delta.days // 365
        months = (time_delta.days % 365) // 30
        days = (time_delta.days % 365) % 30
        print(f'  Duração total aproximada: {years} Anos, {months} Meses, {days} Dias')
    else:
        print("Série está vazia.")
        return

    # Agregações temporais para plot (contagem de ocorrências se for uma série de eventos, ou média/soma se forem valores)
    # Assumindo que a série contém valores e queremos a média por período, ou contagem se for uma série de timestamps de eventos.
    # Para generalizar, vamos usar value_counts() se a série for de timestamps distintos, ou resample().mean() se for de valores.
    # O código original usava value_counts, implicando frequência de timestamps. Vamos manter essa lógica.

    # Frequência diária
    daily_freq = series_datetime_idx.index.normalize().value_counts().sort_index()
    daily_freq = daily_freq.reindex(pd.date_range(daily_freq.index.min(), daily_freq.index.max(), freq='D'),
                                    fill_value=0)
    daily_moving_avg = daily_freq.rolling(window=30, center=True, min_periods=1).mean()

    # Frequência mensal
    monthly_freq = series_datetime_idx.groupby(series_datetime_idx.index.to_period('M')).size()
    monthly_freq.index = monthly_freq.index.to_timestamp()  # Converte PeriodIndex para DatetimeIndex para plot
    monthly_freq = monthly_freq.sort_index()

    # Frequência anual
    yearly_freq = series_datetime_idx.groupby(series_datetime_idx.index.year).size()
    yearly_freq.index = pd.to_datetime(yearly_freq.index, format='%Y')  # Converte índice de ano para DatetimeIndex
    yearly_freq = yearly_freq.sort_index()

    # Plots de Frequência
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=False)  # sharex=False pois as escalas de tempo são diferentes

    # Plot Diário
    ax[0].plot(daily_freq.index, daily_freq, label='Frequência Diária', color=main_color, alpha=0.7)
    ax[0].plot(daily_moving_avg.index, daily_moving_avg, label='Média Móvel (30 dias)', color='firebrick',
               linestyle='--')
    ax[0].set_title(f'Distribuição de "{series_name}" por Dia', fontsize='medium')
    ax[0].set_ylabel('Frequência')
    ax[0].legend(fontsize='small')
    ax[0].grid(True, linestyle=':', alpha=0.6)

    # Plot Mensal (Barras)
    ax[1].bar(monthly_freq.index, monthly_freq, width=20, label='Frequência Mensal', color=main_color,
              alpha=0.8)  # width ajustado para barras mensais
    ax[1].set_title(f'Distribuição de "{series_name}" por Mês', fontsize='medium')
    ax[1].set_ylabel('Frequência')
    ax[1].xaxis.set_major_formatter(plt.FixedFormatter(monthly_freq.index.strftime('%b/%Y')))  # Formato Mês/Ano
    ax[1].tick_params(axis='x', rotation=45, ha='right')
    ax[1].grid(True, linestyle=':', alpha=0.6, axis='y')

    # Plot Anual (Barras)
    ax[2].bar(yearly_freq.index, yearly_freq, width=300, label='Frequência Anual', color=main_color,
              alpha=0.9)  # width ajustado para barras anuais
    ax[2].set_title(f'Distribuição de "{series_name}" por Ano', fontsize='medium')
    ax[2].set_xlabel('Tempo')
    ax[2].set_ylabel('Frequência')
    ax[2].xaxis.set_major_formatter(plt.FixedFormatter(yearly_freq.index.strftime('%Y')))  # Formato Ano
    ax[2].grid(True, linestyle=':', alpha=0.6, axis='y')

    plt.tight_layout()
    plt.show()

    if show_stats_plots:
        series_for_stats = daily_freq.copy()  # Usa a frequência diária para análise de estacionaridade e ACF/PACF
        if series_for_stats.empty:
            print("Série de frequência diária está vazia. Análise estatística não pode prosseguir.")
            return

        print(f'\n--- Análise Estatística da Frequência Diária de "{series_name}" ---')

        # Teste Dickey-Fuller
        try:
            adf_p_value = sm.tsa.stattools.adfuller(series_for_stats)[1]
            print(f'Teste Dickey-Fuller (para frequência diária): P-valor = {adf_p_value:.5f}')
            if adf_p_value <= 0.05:
                print("  Conclusão: Provavelmente estacionária.")
            else:
                print("  Conclusão: Provavelmente não-estacionária.")
        except Exception as e:
            print(f"  Erro no teste Dickey-Fuller: {e}")

        # Plots estatísticos: Série com Média Móvel, ACF, PACF
        fig_stats, ((ax_ts), (ax_acf), (ax_pacf)) = plt.subplots(3, 1, figsize=(15, 12),
                                                                 gridspec_kw={'height_ratios': [2, 1,
                                                                                                1]})  # Ajusta altura dos subplots

        ax_ts.plot(series_for_stats.index, series_for_stats, color=main_color, label='Frequência Diária', alpha=0.7)
        ax_ts.plot(daily_moving_avg.index, daily_moving_avg, color='firebrick', label='Média Móvel (30 dias)',
                   linestyle='--')
        ax_ts.set_title(
            f'Análise da Série Temporal (Frequência Diária de "{series_name}")\nTeste Dickey-Fuller: p={adf_p_value:.4f}',
            fontsize='medium')
        ax_ts.legend(fontsize='small')
        ax_ts.set_ylabel('Frequência')
        ax_ts.grid(True, linestyle=':', alpha=0.6)

        try:
            plot_acf(series_for_stats, ax=ax_acf, lags=min(40, len(series_for_stats) // 2 - 1),
                     fft=False)  # fft=False se houver NaNs (já tratados)
            ax_acf.set_title('Função de Autocorrelação (ACF)', fontsize='small')
            ax_acf.grid(True, linestyle=':', alpha=0.6)
        except Exception as e:
            ax_acf.set_title(f'Erro ao plotar ACF: {e}', fontsize='small', color='red')

        try:
            plot_pacf(series_for_stats, ax=ax_pacf, lags=min(40, len(series_for_stats) // 2 - 1), method='ols')
            ax_pacf.set_title('Função de Autocorrelação Parcial (PACF)', fontsize='small')
            ax_pacf.grid(True, linestyle=':', alpha=0.6)
            ax_pacf.set_xlabel('Lag')
        except Exception as e:
            ax_pacf.set_title(f'Erro ao plotar PACF: {e}', fontsize='small', color='red')

        plt.tight_layout()
        plt.show()


def analyze_outliers_iqr(series: pd.Series, multiplier=1.5, verbose=True):
    """
    Identifica e opcionalmente remove outliers de uma série numérica usando o método IQR (Interquartile Range).

    Parâmetros:
    series (pd.Series): Série de valores numéricos.
    multiplier (float, opcional): Multiplicador do IQR para definir os limites.
                                  Valores comuns são 1.5 (para outliers "leves") ou 3.0 (para outliers "extremos").
                                  Padrão: 1.5.
    verbose (bool, opcional): Se True, imprime informações sobre os outliers encontrados. Padrão: True.

    Retorna:
    pd.Series: Uma nova série com outliers substituídos por np.nan.
               Se nenhum outlier for encontrado, retorna uma cópia da série original.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_value = q3 - q1

    lower_limit = q1 - (iqr_value * multiplier)
    upper_limit = q3 + (iqr_value * multiplier)

    # Identifica outliers
    outliers_mask = (series < lower_limit) | (series > upper_limit)
    num_outliers = outliers_mask.sum()

    if verbose:
        print(f"--- Análise de Outliers (Método IQR, Multiplicador={multiplier}) para '{series.name or "Série"}' ---")
        print(f"  Q1 (25º Percentil): {q1:.4f}")
        print(f"  Q3 (75º Percentil): {q3:.4f}")
        print(f"  IQR (Intervalo Interquartil): {iqr_value:.4f}")
        print(f"  Limite Inferior para Outliers: {lower_limit:.4f}")
        print(f"  Limite Superior para Outliers: {upper_limit:.4f}")
        print(f"  Número de Outliers Encontrados: {num_outliers} ({num_outliers * 100 / len(series):.2f}% do total)")
        if num_outliers == 0:
            print("  Nenhum outlier detectado com os critérios atuais.")
        else:
            print(f"  Valores considerados outliers: {series[outliers_mask].tolist()[:10]} (mostrando até 10)")

    # Retorna a série com outliers como NaN (o código original fazia isso implicitamente)
    series_no_outliers = series.copy()
    series_no_outliers[outliers_mask] = np.nan

    return series_no_outliers


def plot_numeric_variable_distribution(series: pd.Series, main_color='steelblue', remove_outliers_iqr=True,
                                       iqr_multiplier=1.5):
    """
    Cria um painel de visualização para uma variável numérica.
    Inclui estatísticas descritivas, violin plot com boxplot sobreposto e histograma.
    Permite a remoção opcional de outliers via método IQR antes do plot.

    Parâmetros:
    series (pd.Series): Série numérica a ser analisada.
    main_color (str, opcional): Cor principal para os gráficos. Padrão: 'steelblue'.
    remove_outliers_iqr (bool, opcional): Se True, remove outliers usando o método IQR antes de plotar
                                          e calcular estatísticas (exceto as de outliers). Padrão: True.
    iqr_multiplier (float, opcional): Multiplicador para o método IQR, se `remove_outliers_iqr` for True.
                                      Padrão: 1.5.
    """
    if not pd.api.types.is_numeric_dtype(series):
        print(f"Erro: A série '{series.name}' não é numérica. Esta função é para variáveis numéricas.")
        return

    series_to_plot = series.copy()
    series_name = series.name or "Variável Numérica"

    if remove_outliers_iqr:
        # A função analyze_outliers_iqr já imprime detalhes sobre os outliers
        series_no_outliers = analyze_outliers_iqr(series_to_plot, multiplier=iqr_multiplier, verbose=True)
        # Para os plots e estatísticas (exceto skew/kurtosis que podem ser sensíveis), usa a série sem outliers
        series_for_plots_stats = series_no_outliers.dropna()
    else:
        series_for_plots_stats = series_to_plot.dropna()

    if series_for_plots_stats.empty:
        print(f"A série '{series_name}' está vazia após o tratamento de NaNs/outliers. Nada para plotar.")
        return

    # Estatísticas Descritivas
    desc_stats = series_for_plots_stats.describe().to_frame().T  # Transpõe para melhor visualização
    # Adiciona Skewness e Kurtosis (calculadas na série com/sem outliers conforme a escolha)
    try:
        desc_stats['skewness'] = skew(series_for_plots_stats)
        desc_stats['kurtosis'] = kurtosis(series_for_plots_stats)  # Fisher's kurtosis (normal = 0)
    except Exception as e:
        print(f"Aviso: Não foi possível calcular skewness/kurtosis: {e}")
        desc_stats['skewness'] = np.nan
        desc_stats['kurtosis'] = np.nan

    print(
        f"\n--- Estatísticas Descritivas para '{series_name}' {' (após remoção de outliers)' if remove_outliers_iqr else ''} ---")
    print(desc_stats.to_string())

    # Configuração dos Plots
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # Violin Plot com Boxplot sobreposto
    sns.violinplot(x=series_for_plots_stats, color=main_color, inner=None, ax=ax[0],
                   linewidth=0)  # inner=None para remover barras internas
    # Sobrepõe um boxplot mais sutil
    sns.boxplot(x=series_for_plots_stats, color='white', ax=ax[0], width=0.3,
                boxprops=dict(edgecolor='black', facecolor=(0, 0, 0, 0)),  # Caixa transparente com borda
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='red'))
    # Ajusta a opacidade do violin para ver o boxplot
    for collection in ax[0].collections:
        if isinstance(collection, plt.cm.ScalarMappable):  # Evita erro se não for PolyCollection
            pass
        else:
            collection.set_alpha(0.4)

    ax[0].set_xlabel('')  # Remove xlabel do subplot superior, pois é compartilhado
    ax[0].set_ylabel(series_name.title().replace('_', ' '), fontsize=11)
    ax[0].set_title(f'Distribuição (Violin & Box Plot) de: {series_name}', fontsize=13)

    # Histograma com KDE
    sns.histplot(x=series_for_plots_stats, color=main_color, ax=ax[1], kde=True)
    ax[1].axvline(series_for_plots_stats.mean(), color='red', linestyle='--', linewidth=1.5,
                  label=f'Média: {series_for_plots_stats.mean():.2f}')
    ax[1].axvline(series_for_plots_stats.median(), color='black', linestyle=':', linewidth=1.5,
                  label=f'Mediana: {series_for_plots_stats.median():.2f}')
    ax[1].set_xlabel('Valor', fontsize=11)
    ax[1].set_ylabel('Frequência', fontsize=11)
    ax[1].set_title(f'Histograma de: {series_name}', fontsize=13)
    ax[1].legend(fontsize='small')
    ax[1].grid(True, linestyle=':', alpha=0.6, axis='y')

    plt.suptitle(f"Análise da Variável Numérica: {series_name}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_temperature_timeseries_analysis(
        temp_series: pd.Series,
        series_name: str = 'Temperatura',
        window_days: int = 30,  # Janela para média móvel em dias
        main_color: str = 'dodgerblue',
        show_stationarity_analysis: bool = True
):
    """
    Realiza e plota uma análise detalhada de uma série temporal de temperatura.

    Inclui:
    - Plot da série temporal original.
    - Plot da média móvel centrada.
    - (Opcional) Análise de estacionariedade com teste ADF.
    - (Opcional) Gráficos de ACF e PACF.

    Parâmetros:
    temp_series (pd.Series): Série temporal dos dados de temperatura.
                             O índice DEVE ser do tipo pd.DatetimeIndex.
    series_name (str, opcional): Nome da série para usar nos títulos dos gráficos. Padrão: 'Temperatura'.
    window_days (int, opcional): Janela em DIAS para o cálculo da média móvel. Padrão: 30.
    main_color (str, opcional): Cor principal para os gráficos da série original. Padrão: 'dodgerblue'.
    show_stationarity_analysis (bool, opcional): Se True, realiza e exibe a análise de estacionariedade
                                                 (teste ADF, plots ACF/PACF). Padrão: True.
    """
    if not isinstance(temp_series.index, pd.DatetimeIndex):
        print("Erro Crítico: O índice da série de temperatura DEVE ser do tipo DatetimeIndex.")
        try:  # Tenta converter
            temp_series_proc = temp_series.copy()
            temp_series_proc.index = pd.to_datetime(temp_series_proc.index)
            print("Índice convertido para DatetimeIndex. Prosseguindo com a análise.")
        except Exception as e:
            print(f"Falha ao converter o índice para DatetimeIndex: {e}. Análise interrompida.")
            return
    else:
        temp_series_proc = temp_series.copy()  # Usa cópia para evitar modificar original

    if temp_series_proc.empty:
        print(f"Erro: A série temporal '{series_name}' fornecida está vazia.")
        return

    print(f"\n--- Análise Detalhada da Série Temporal: {series_name} ---")
    print(f"Período dos dados: {temp_series_proc.index.min()} a {temp_series_proc.index.max()}")
    print(f"Número de observações: {len(temp_series_proc)}")
    nan_count_original = temp_series_proc.isnull().sum()
    print(f"Valores ausentes (NaN) na série original: {nan_count_original}")

    # Determina a frequência dos dados para calcular a janela em número de observações
    # Tenta inferir a frequência. Se não conseguir, assume uma frequência comum (ex: horária ou diária)
    # ou pede ao usuário. Para o dataset Jena, os dados são a cada 10 minutos.
    inferred_freq = pd.infer_freq(temp_series_proc.index)
    obs_per_day = None
    if inferred_freq:
        if 'T' in inferred_freq or 'min' in inferred_freq:  # Ex: '10T' para 10 minutos
            try:
                minutes_interval = int(inferred_freq.replace('T', '').replace('min', ''))
                obs_per_day = (24 * 60) / minutes_interval
            except:
                pass  # Ignora se não conseguir parsear
        elif 'H' in inferred_freq:  # Ex: 'H' para horária
            obs_per_day = 24
        elif 'D' in inferred_freq:  # Ex: 'D' para diária
            obs_per_day = 1

    if obs_per_day is None:
        # Fallback se a frequência não puder ser inferida ou não for uma das esperadas
        # Para o dataset Jena (10 min), obs_per_day = 144
        # Se a série for diária, obs_per_day = 1
        # Se a série for horária, obs_per_day = 24
        # Vamos assumir uma frequência de 10 minutos como no dataset original do problema
        print("Aviso: Não foi possível inferir a frequência exata dos dados ou não é padrão (min, H, D). "
              "Assumindo dados a cada 10 minutos para cálculo da janela da média móvel (144 obs/dia).")
        obs_per_day = 144  # 6 observações por hora * 24 horas

    window_obs = int(window_days * obs_per_day)
    if window_obs < 2:  # Mínimo para rolling window
        print(f"Aviso: Janela calculada ({window_obs} observações) é muito pequena. Ajustando para o mínimo de 2.")
        window_obs = 2

    # Calcula média móvel centrada
    # min_periods=1 para calcular mesmo nas bordas onde a janela completa não está disponível
    moving_avg = temp_series_proc.rolling(window=window_obs, center=True, min_periods=1).mean()

    # Plot principal: Série Temporal e Média Móvel
    plt.figure(figsize=(18, 7))
    plt.plot(temp_series_proc.index, temp_series_proc, label=f'{series_name} (Original)', color=main_color, alpha=0.7,
             linewidth=1)
    plt.plot(moving_avg.index, moving_avg, label=f'Média Móvel ({window_days} dias, centrada)', color='firebrick',
             linestyle='-', linewidth=1.5)
    plt.title(f'Série Temporal: {series_name} e Média Móvel ({window_days} dias)', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel(series_name, fontsize=12)
    plt.legend(fontsize='medium')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    if show_stationarity_analysis:
        # Usa a série original (após tratar NaNs) para os testes estatísticos
        series_for_stats = temp_series_proc.dropna()

        if series_for_stats.empty:
            print(
                f"A série '{series_name}' está vazia após remover NaNs. Análise de estacionaridade não pode prosseguir.")
            return

        print(f"\n--- Análise de Estacionaridade para '{series_name}' (após remover {nan_count_original} NaNs) ---")

        # Teste de Estacionaridade (Augmented Dickey-Fuller)
        try:
            adf_result = adfuller(series_for_stats)
            print(f'Teste Augmented Dickey-Fuller (ADF):')
            print(f'  Estatística ADF: {adf_result[0]:.4f}')
            print(f'  P-valor: {adf_result[1]:.4f}')
            print(f'  Lags Usados: {adf_result[2]}')
            print(f'  Número de Observações: {adf_result[3]}')
            print(f'  Valores Críticos:')
            for key, value in adf_result[4].items():
                print(f'    {key}: {value:.4f}')

            if adf_result[1] <= 0.05:
                print(f"  Conclusão (ADF, alpha=0.05): A série '{series_name}' é provavelmente ESTACIONÁRIA.")
            else:
                print(f"  Conclusão (ADF, alpha=0.05): A série '{series_name}' é provavelmente NÃO-ESTACIONÁRIA.")
        except Exception as e:
            print(f"  Erro ao executar o teste ADF: {e}")

        # Plots de ACF e PACF
        # Determina um número razoável de lags para visualização
        num_lags_plot = min(50, len(series_for_stats) // 2 - 1)  # Ex: até 50 lags, ou metade dos dados

        if num_lags_plot < 1:
            print(
                f"  Não há lags suficientes para os gráficos ACF/PACF para '{series_name}' (dados: {len(series_for_stats)}).")
        else:
            fig_stats, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(16, 10))  # Dois subplots verticais

            try:
                plot_acf(series_for_stats, ax=ax_acf, lags=num_lags_plot,
                         title=f'Função de Autocorrelação (ACF) - {series_name}', fft=False)
                ax_acf.grid(True, linestyle=':', alpha=0.6)
            except Exception as e:
                print(f"  Erro ao plotar ACF: {e}")
                ax_acf.set_title(f'Erro ao plotar ACF para {series_name}', color='red')

            try:
                plot_pacf(series_for_stats, ax=ax_pacf, lags=num_lags_plot, method='ols',
                          title=f'Função de Autocorrelação Parcial (PACF) - {series_name}')
                ax_pacf.grid(True, linestyle=':', alpha=0.6)
                ax_pacf.set_xlabel("Lag")  # Adiciona label apenas no último
            except Exception as e:
                print(f"  Erro ao plotar PACF: {e}")
                ax_pacf.set_title(f'Erro ao plotar PACF para {series_name}', color='red')

            plt.tight_layout()
            plt.show()
    print(f"--- Fim da Análise para '{series_name}' ---")
