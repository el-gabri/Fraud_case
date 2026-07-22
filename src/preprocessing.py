import numpy as np
import pandas as pd

FLOAT_COLS = [
    'p (mbar)', 'T (degC)', 'Tdew (degC)',
    'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    'wd (deg)'
]


def prepare_data(df, period="1h", float_cols_list=None):
    """
    Prepara e processa o DataFrame de entrada contendo dados climáticos.

    Esta função realiza as seguintes etapas:
    1. Renomeia as colunas para um formato padronizado.
    2. Remove linhas duplicadas.
    3. Remove colunas desnecessárias (ex: 'Tpot (K)').
    4. Converte as colunas especificadas em `float_cols_list` para o tipo float.
       Se `float_cols_list` não for fornecido, utiliza a lista global `FLOAT_COLS`.
    5. Converte a coluna 'Date Time' para o formato datetime.
    6. Define 'Date Time' como o índice do DataFrame e o ordena.
    7. Imputa valores ausentes (-9999.0) nas colunas de velocidade do vento ('wv (m/s)', 'max. wv (m/s)')
       utilizando a mediana respectiva de cada coluna.
    8. Reamostra os dados para a frequência temporal especificada (`period`), calculando a média
       das colunas numéricas.
    9. Cria novas features baseadas no tempo: 'days_since_beginning' (dias desde a primeira data)
       e 'year' (ano).

    Parâmetros:
    df (pandas.DataFrame): O DataFrame de entrada com os dados climáticos.
                           Espera-se que a primeira coluna seja a data/hora.
    period (str, opcional): O período para reamostragem dos dados (ex: "1h", "10min", "1D").
                            O padrão é "1h" (1 hora).
    float_cols_list (list, opcional): Lista de nomes de colunas que devem ser tratadas como float
                                      e usadas na reamostragem. Se None, a função utilizará
                                      a lista global `FLOAT_COLS`.

    Retorna:
    pandas.DataFrame: Um DataFrame processado e reamostrado, com as colunas numéricas
                      agregadas pela média ao longo do período especificado.
                      Retorna None se ocorrer um erro crítico.
    """
    # Faz uma cópia para evitar modificar o DataFrame original passado à função
    df_processed = df.copy()

    # Define os nomes esperados para as colunas.
    expected_columns = ['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
                        'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
                        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
                        'wd (deg)']
    if len(df_processed.columns) == len(expected_columns):
        df_processed.columns = expected_columns
        print("Colunas renomeadas com sucesso.")
    else:
        print(f"Aviso: O número de colunas no DataFrame ({len(df_processed.columns)}) "
              f"não corresponde ao número esperado ({len(expected_columns)}). "
              "Verifique a estrutura do arquivo CSV. Continuando com os nomes originais se possível.")

    # Remove linhas duplicadas
    df_processed.drop_duplicates(inplace=True)
    print(f"Linhas duplicadas removidas. Shape atual: {df_processed.shape}")

    # Remove a coluna 'Tpot (K)' (Temperatura em Kelvin), pois é redundante com 'T (degC)'
    if "Tpot (K)" in df_processed.columns:
        df_processed.drop("Tpot (K)", axis=1, inplace=True)
        print("Coluna 'Tpot (K)' removida.")
    else:
        print("Aviso: Coluna 'Tpot (K)' não encontrada para remoção.")

    # Define a lista de colunas a serem convertidas para float
    # Se float_cols_list não for fornecido, usa a lista global FLOAT_COLS
    cols_to_convert_to_float = float_cols_list if float_cols_list is not None else FLOAT_COLS

    # Garante que apenas colunas existentes no DataFrame sejam processadas
    actual_float_cols = [col for col in cols_to_convert_to_float if col in df_processed.columns]

    if len(actual_float_cols) != len(cols_to_convert_to_float):
        missing_cols = set(cols_to_convert_to_float) - set(actual_float_cols)
        print(
            f"Aviso: As seguintes colunas especificadas em `float_cols_list` (ou `FLOAT_COLS`) não foram encontradas e serão ignoradas: {missing_cols}")

    if not actual_float_cols:
        print(
            "Erro: Nenhuma coluna float válida foi identificada para processamento. Verifique `float_cols_list` ou os nomes das colunas.")
        return None

    # Converte as colunas identificadas para o tipo float
    try:
        df_processed[actual_float_cols] = df_processed[actual_float_cols].astype(float)
        print(f"Colunas {actual_float_cols} convertidas para o tipo float.")
    except ValueError as e:
        print(f"Erro ao converter colunas para float: {e}. Verifique os dados nessas colunas.")
        return None

    # Converte a coluna 'Date Time' para o formato datetime
    if 'Date Time' in df_processed.columns:
        try:
            df_processed['Date Time'] = pd.to_datetime(df_processed['Date Time'], format='%d.%m.%Y %H:%M:%S')
            print("Coluna 'Date Time' convertida para datetime com sucesso.")
        except ValueError as e:
            print(f"Erro crítico ao converter 'Date Time': {e}.")
            print("Verifique o formato da data na coluna 'Date Time'.")
            return None
    else:
        print("Erro crítico: Coluna 'Date Time' não encontrada.")
        return None

    # Define 'Date Time' como índice e ordena
    df_processed.set_index('Date Time', inplace=True)
    df_processed.sort_index(inplace=True)
    print("Coluna 'Date Time' definida como índice e ordenada.")

    # Imputação de valores -9999.0 para velocidade do vento
    wind_cols_to_impute = ['wv (m/s)', 'max. wv (m/s)']
    for col in wind_cols_to_impute:
        if col in df_processed.columns:
            # Substitui -9999 por NaN para calcular a mediana corretamente
            median_val = df_processed[col].replace(-9999.0, np.nan).median()
            print(f"Mediana calculada para '{col}' (ignorando -9999): {median_val}")

            df_processed[col] = df_processed[col].replace(-9999.0, median_val)
            print(f"Valores -9999.0 em '{col}' substituídos pela mediana ({median_val}).")

            if df_processed[col].isnull().any():
                print(f"Aviso: Após imputação, '{col}' ainda contém NaNs (mediana pode ter sido NaN).")
                if pd.isna(median_val):
                    df_processed[col].fillna(0, inplace=True)  # Preenche NaNs restantes com 0 se mediana foi NaN
                    print(f"NaNs restantes em '{col}' preenchidos com 0.")
        else:
            print(f"Aviso: Coluna '{col}' não encontrada para imputação de -9999.0.")

    # Reamostragem dos dados
    if period is not None:
        try:
            # Usa apenas as colunas que foram efetivamente convertidas para float e existem
            df_resample = df_processed[actual_float_cols].resample(period).mean()
            print(f"Dados reamostrados para o período '{period}' usando a média.")
        except Exception as e:
            print(f"Erro durante a reamostragem dos dados: {e}")
            return None
    else:
        df_resample = df.copy()
        print("Os dados não foram reamostrados.")

    df_resample.reset_index(inplace=True)  # Transforma o índice 'Date Time' de volta em coluna

    # Engenharia de features baseadas no tempo
    if not df_resample.empty and 'Date Time' in df_resample.columns:
        # Dias desde o início
        df_resample["days_since_beginning"] = (
                df_resample["Date Time"].dt.normalize() - df_resample["Date Time"].dt.normalize().min()
        ).dt.days

        df_resample["month"] = df_resample["Date Time"].dt.month
        df_resample["year"] = df_resample["Date Time"].dt.year
        print("Features 'days_since_beginning' e 'year' criadas.")
    else:
        print("Aviso: DataFrame reamostrado está vazio ou 'Date Time' não é coluna. Features de tempo não criadas.")

    return df_resample


def _get_seasonality(series_length, period=365.25, phase_shift=0, amplitude=1):
    """
    Gera componentes de sazonalidade (seno e cosseno) para modelar ciclos periódicos.

    Parâmetros:
    series_length (int): O número total de observações (passos de tempo).
    period (float, opcional): Número de passos de tempo para um ciclo completo
                              (ex: 365.25 para anual com dados diários, 24 para diária com dados horários).
                              Padrão é 365.25.
    phase_shift (float, opcional): Deslocamento de fase em radianos. Padrão é 0.
    amplitude (float, opcional): Amplitude das ondas. Padrão é 1.

    Retorna:
    pandas.DataFrame: DataFrame com colunas 'date_cos' e 'date_sin'.
    """
    if not isinstance(series_length, int) or series_length <= 0:
        raise ValueError("O parâmetro 'series_length' deve ser um inteiro positivo.")
    if not isinstance(period, (int, float)) or period <= 0:
        raise ValueError("O parâmetro 'period' deve ser um número positivo.")

    time_steps = np.arange(0, series_length)
    df_seasonality = pd.DataFrame(index=pd.RangeIndex(series_length))  # Garante mesmo comprimento de índice

    # Componente cossenoidal
    df_seasonality["date_cos"] = np.cos(2 * np.pi * time_steps / period + phase_shift) * amplitude
    # Componente senoidal
    df_seasonality["date_sin"] = np.sin(2 * np.pi * time_steps / period + phase_shift) * amplitude
    return df_seasonality


def add_seasonal_features(df, date_col_name="Date Time"):
    """
    Adiciona features temporais sazonais (anuais, mensais, diárias) ao DataFrame.

    Assume-se que o DataFrame de entrada `df` possui uma frequência horária,
    ou seja, cada linha representa uma hora, para que os períodos definidos façam sentido.

    Parâmetros:
    df (pandas.DataFrame): DataFrame de entrada.
    date_col_name (str, opcional): Nome da coluna de data/hora. Não é diretamente usada para
                                   cálculos nesta versão, mas mantida para semântica.
                                   O padrão é "Date Time".

    Retorna:
    pandas.DataFrame: Novo DataFrame com features sazonais adicionadas.
                      Retorna None se o DataFrame de entrada estiver vazio.
    """
    if df.empty:
        print("Erro: DataFrame de entrada está vazio. Não é possível adicionar features sazonais.")
        return None

    df_copy = df.copy()
    series_len = len(df_copy)

    # Períodos em número de horas (assumindo dados horários)
    daily_period_hours = 24
    monthly_period_hours = daily_period_hours * 30.4375  # Média de dias por mês
    yearly_period_hours = daily_period_hours * 365.25  # Considera anos bissextos

    # Gera componentes sazonais
    sazonality_year = _get_seasonality(series_len, period=yearly_period_hours)
    sazonality_month = _get_seasonality(series_len, period=monthly_period_hours)
    sazonality_day = _get_seasonality(series_len, period=daily_period_hours)

    # Renomeia colunas
    sazonality_year.columns = ["year_cos", "year_sin"]
    sazonality_month.columns = ["month_cos", "month_sin"]
    sazonality_day.columns = ["day_cos", "day_sin"]

    # Concatena os componentes sazonais.
    # Garante que os índices sejam compatíveis para a concatenação.
    sazonality_year.index = df_copy.index
    sazonality_month.index = df_copy.index
    sazonality_day.index = df_copy.index

    seasonal_components_df = pd.concat(
        [sazonality_year, sazonality_month, sazonality_day], axis=1
    )

    # Adiciona ao DataFrame original (cópia)
    df_featured = pd.concat([df_copy, seasonal_components_df], axis=1)

    print("Features sazonais (anuais, mensais, diárias) adicionadas.")
    return df_featured


def add_lag_features(df, feature_cols_to_lag, lags_list=[24, 48, 72]):
    """
    Adiciona features defasadas (lagged features) ao DataFrame.

    Assume-se que `lags_list` contém defasagens em número de linhas/observações.
    Se os dados são horários, um lag de 24 significa 24 horas.

    Parâmetros:
    df (pandas.DataFrame): DataFrame de entrada.
    feature_cols_to_lag (list): Lista de nomes de colunas para criar features defasadas.
    lags_list (list, opcional): Lista de valores de defasagem. Padrão é [24, 48, 72].

    Retorna:
    pandas.DataFrame: Novo DataFrame com features defasadas.
    """

    df_copy = df.copy()

    for lag_value in lags_list:
        for col_name in feature_cols_to_lag:
            if col_name in df_copy.columns:
                df_copy[f"{col_name}_lag_{lag_value}h"] = df_copy[col_name].shift(lag_value)
            else:
                print(f"Aviso: Coluna '{col_name}' para lag não encontrada. Será ignorada para lag {lag_value}h.")

    print(f"Features de lag adicionadas para os lags: {lags_list}.")
    return df_copy


def add_moving_average_features(df, feature_cols_for_ma, window_size=24):
    """
    Adiciona features de média móvel ao DataFrame.

    Assume-se que `window_size` é em número de linhas/observações.
    Se os dados são horários, window_size=24 calcula a média das últimas 24 horas.

    Parâmetros:
    df (pandas.DataFrame): DataFrame de entrada.
    feature_cols_for_ma (list): Lista de nomes de colunas para calcular médias móveis.
    window_size (int, opcional): Tamanho da janela. Padrão é 24.

    Retorna:
    pandas.DataFrame: Novo DataFrame com features de média móvel.
    """
    df_copy = df.copy()

    for col_name in feature_cols_for_ma:
        if col_name in df_copy.columns:
            df_copy[f"{col_name}_mean_{str(window_size)}h"] = df_copy[col_name].rolling(
                window=window_size, min_periods=1  # min_periods=1 calcula mesmo com menos dados no início
            ).mean()
        else:
            print(f"Aviso: Coluna '{col_name}' para média móvel não encontrada. Será ignorada.")

    print(f"Features de média móvel (janela de {window_size}h) adicionadas.")
    return df_copy


def remove_null_rows(df, feature_columns, target_column_name):
    """
    Remove linhas do DataFrame que contêm valores nulos nas colunas de features
    especificadas ou na coluna alvo.

    Parâmetros:
    df (pandas.DataFrame): DataFrame de entrada.
    feature_columns (list): Lista de nomes das colunas de features.
    target_column_name (str): Nome da coluna alvo.

    Retorna:
    pandas.DataFrame: Novo DataFrame sem as linhas que continham valores nulos
                      nas colunas especificadas.
    """
    df_copy = df.copy()  # Trabalha com uma cópia para não modificar o original

    # Constrói a lista de colunas para verificar NaNs
    cols_to_check_for_nan = []

    # Adiciona colunas de features que existem no DataFrame
    for col in feature_columns:
        if col in df_copy.columns:
            cols_to_check_for_nan.append(col)
        else:
            print(f"Aviso: Coluna de feature '{col}' para checagem de nulos não encontrada no DataFrame.")

    # Adiciona a coluna alvo se ela existir no DataFrame
    if target_column_name in df_copy.columns:
        cols_to_check_for_nan.append(target_column_name)
    else:
        print(f"Aviso: Coluna alvo '{target_column_name}' para checagem de nulos não encontrada no DataFrame.")

    if not cols_to_check_for_nan:
        print(
            "Aviso: Nenhuma coluna válida (features ou alvo) foi encontrada para verificar nulos. Retornando DataFrame original.")
        return df_copy

    # Remove linhas onde QUALQUER UMA das colunas em `cols_to_check_for_nan` seja NaN.
    df_cleaned = df_copy.dropna(subset=cols_to_check_for_nan, axis=0, how="any")

    rows_dropped = len(df_copy) - len(df_cleaned)
    if rows_dropped > 0:
        print(
            f"{rows_dropped} linhas contendo valores nulos foram removidas com base nas colunas: {cols_to_check_for_nan}.")
    else:
        print(f"Nenhuma linha continha valores nulos nas colunas especificadas: {cols_to_check_for_nan}.")

    return df_cleaned
