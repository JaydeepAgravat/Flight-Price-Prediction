import pandas as pd    

def clean_column_names(df):
    # Convert all column names to lowercase for consistency
    return df.rename(columns=str.lower)

def strip_string_columns(df):
    # Identify columns with string data type
    string_columns = df.select_dtypes(include='O').columns
    # Strip leading and trailing whitespace from all string columns
    for col in string_columns:
        df[col] = df[col].str.strip()
    return df

def clean_airline_names(df):
    # Clean and standardize 'airline' column by removing specific substrings and title-casing
    df['airline'] = (
        df['airline']
        .str.replace(" Premium economy", "")
        .str.replace(" Business", "")
        .str.title()
    )
    return df

def convert_dates(df):
    # Convert 'date_of_journey' column to datetime format, assuming day-first format
    df['date_of_journey'] = pd.to_datetime(df['date_of_journey'], dayfirst=True)
    return df

def convert_times(df):
    # Convert 'dep_time' and 'arrival_time' columns to time format
    df['dep_time'] =  pd.to_datetime(df['dep_time'], format='mixed').dt.time
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='mixed').dt.time
    return df

def convert_duration(df):
    # Split 'duration' column into hours and minutes
    duration_split = df['duration'].str.split(" ", expand=True).set_axis(["hour", "minute"], axis=1)
    # Convert hours to minutes and fill missing minutes with 0
    duration_split['hour'] = duration_split['hour'].str.replace("h", "").astype(int).mul(60)
    duration_split['minute'] = duration_split['minute'].str.replace("m", "").fillna("0").astype(int)
    # Sum hours and minutes to get total duration in minutes
    df['duration_minute'] = duration_split.sum(axis=1)
    # drop duration column
    df = df.drop(columns=['duration'])
    return df

def convert_total_stops(df):
    # Standardize and convert 'total_stops' column to numeric
    df['total_stops'] = (
        df['total_stops']
        .replace("non-stop", "0")  # Replace 'non-stop' with '0'
        .str.replace(" stops?", "", regex=True)  # Remove ' stop' or ' stops'
        .pipe(lambda ser: pd.to_numeric(ser))  # Convert to numeric
    )
    return df

def lower_additional_info(df):
    # Convert 'additional_info' column to lowercase
    df['additional_info'] = df['additional_info'].str.lower()
    return df

def clean_df(df):
    # Drop rows where 'Duration' is '5m' and drop the 'Route' column
    df = df.drop(index=df[df['Duration'].isin(['5m'])].index, columns=['Route'])
    # Drop duplicate rows to ensure data consistency
    df = df.drop_duplicates()
    # Drop null values
    df = df.dropna()
    # Apply all preprocessing steps in sequence to clean and standardize the DataFrame
    df = clean_column_names(df)         # Convert column names to lowercase
    df = strip_string_columns(df)       # Strip leading/trailing whitespace from string columns
    df = clean_airline_names(df)        # Clean and standardize 'airline' column
    df = convert_dates(df)              # Convert 'date_of_journey' column to datetime format
    df = convert_times(df)              # Convert 'dep_time' and 'arrival_time' columns to time format
    df = convert_duration(df)           # Convert 'duration' column to total minutes
    df = convert_total_stops(df)        # Standardize and convert 'total_stops' column to numeric
    df = lower_additional_info(df)      # Convert 'additional_info' column to lowercase
    # Drop duplicate rows to ensure data consistency
    df = df.drop_duplicates()
    df = df.dropna()
    return df


