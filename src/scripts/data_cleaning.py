import pandas as pd    

def clean_column_names(df):
    """
    Convert all column names to lowercase for consistency.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
    """
    try:
        return df.rename(columns=str.lower)
    except Exception as e:
        print(f"An error occurred while cleaning column names: {e}")
        return df

def strip_string_columns(df):
    """
    Strip leading and trailing whitespace from all string columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with stripped string columns.
    """
    try:
        string_columns = df.select_dtypes(include='O').columns
        for col in string_columns:
            df[col] = df[col].str.strip()
        return df
    except Exception as e:
        print(f"An error occurred while stripping string columns: {e}")
        return df

def clean_airline_names(df):
    """
    Clean and standardize 'airline' column by removing specific substrings and title-casing.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with cleaned 'airline' column.
    """
    try:
        df['airline'] = (
            df['airline']
            .str.replace(" Premium economy", "", regex=False)
            .str.replace(" Business", "", regex=False)
            .str.title()
        )
        return df
    except Exception as e:
        print(f"An error occurred while cleaning airline names: {e}")
        return df

def convert_dates(df):
    """
    Convert 'date_of_journey' column to datetime format, assuming day-first format.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with converted 'date_of_journey' column.
    """
    try:
        df['date_of_journey'] = pd.to_datetime(df['date_of_journey'], dayfirst=True)
        return df
    except Exception as e:
        print(f"An error occurred while converting dates: {e}")
        return df

def convert_times(df):
    """
    Convert 'dep_time' and 'arrival_time' columns to time format.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with converted time columns.
    """
    try:
        df['dep_time'] =  pd.to_datetime(df['dep_time'], format='mixed').dt.time
        df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='mixed').dt.time
        return df
    except Exception as e:
        print(f"An error occurred while converting times: {e}")
        return df

def convert_duration(df):
    """
    Convert 'duration' column to total minutes.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with converted 'duration' column.
    """
    try:
        duration_split = df['duration'].str.split(" ", expand=True).set_axis(["hour", "minute"], axis=1)
        duration_split['hour'] = duration_split['hour'].str.replace("h", "").astype(int).mul(60)
        duration_split['minute'] = duration_split['minute'].str.replace("m", "").fillna("0").astype(int)
        df['duration_minute'] = duration_split.sum(axis=1)
        df = df.drop(columns=['duration'])
        return df
    except Exception as e:
        print(f"An error occurred while converting duration: {e}")
        return df

def convert_total_stops(df):
    """
    Standardize and convert 'total_stops' column to numeric.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with converted 'total_stops' column.
    """
    try:
        df['total_stops'] = (
            df['total_stops']
            .replace("non-stop", "0")
            .str.replace(" stops?", "", regex=True)
            .pipe(lambda ser: pd.to_numeric(ser))
        )
        return df
    except Exception as e:
        print(f"An error occurred while converting total stops: {e}")
        return df

def lower_additional_info(df):
    """
    Convert 'additional_info' column to lowercase.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with 'additional_info' column in lowercase.
    """
    try:
        df['additional_info'] = df['additional_info'].str.lower()
        return df
    except Exception as e:
        print(f"An error occurred while converting additional_info: {e}")
        return df

def clean_df(df):
    """
    Apply all preprocessing steps in sequence to clean and standardize the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    try:
        df = df.drop(index=df[df['Duration'].isin(['5m'])].index, columns=['Route'])
        df = df.drop_duplicates()
        df = df.dropna()

        df = clean_column_names(df)
        df = strip_string_columns(df)
        df = clean_airline_names(df)
        df = convert_dates(df)
        df = convert_times(df)
        df = convert_duration(df)
        df = convert_total_stops(df)
        df = lower_additional_info(df)

        df = df.drop_duplicates()
        df = df.dropna()

        return df
    except KeyError as e:
        print(f"Key error during cleaning process: {e}")
        return df
    except Exception as e:
        print(f"An error occurred during the cleaning process: {e}")
        return df
