import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.utils.save_model import save
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer, PowerTransformer
from feature_engine.encoding import RareLabelEncoder, MeanEncoder, CountFrequencyEncoder
from feature_engine.datetime import DatetimeFeatures

# Define transformers for different features
airline_transformer = Pipeline(steps=[
    ('grouper', RareLabelEncoder(tol=0.1, replace_with='other', n_categories=2)),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

date_transformer = Pipeline(steps=[
    ('date_to_features', DatetimeFeatures(features_to_extract=['month', 'day_of_week', 'day_of_year'], yearfirst=True, format='mixed')),
    ('min_max_scaler', MinMaxScaler())
])

location_transformer = Pipeline(steps=[
    ('grouper', RareLabelEncoder(tol=0.1, replace_with='other', n_categories=2)),
    ('mean_encoder', MeanEncoder()),
    ('power_transformer', PowerTransformer())
])

# Define a custom function to determine if a location is in the north
def is_north(X):
    """
    Determine if a location is in the north.
    """
    columns = X.columns.to_list()
    north_cities = {"Delhi", "Kolkata", "Mumbai", "New Delhi"}
    return (
        X
        .assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

location_union_transformer = FeatureUnion(transformer_list=[
    ("location_transformer", location_transformer),
    ("is_north_transformer", FunctionTransformer(func=is_north))
])

time_transformer = Pipeline(
    steps=[
        ('dt', DatetimeFeatures(features_to_extract=['hour', 'minute'], yearfirst=True, format='mixed')),
        ('scaler', MinMaxScaler())
    ]
)

# Define a custom function to determine the part of the day
def part_of_day(X, start=0, mid=8, end=16):
    """
    Determine the part of the day based on hours.
    """
    columns = X.columns.to_list()
    X_temp = X.assign(
        **{
            col: pd.to_datetime(X.loc[:, col]).dt.hour
            for col in columns
        }
    )
    return (
        X_temp
        .assign(
            **{
                f'{col}_part_of_day': np.select(
                    [
                        X_temp.loc[:, col].between(start, mid, inclusive='left'),
                        X_temp.loc[:, col].between(mid, end, inclusive='left')
                    ], choicelist=['start', 'mid'], default='end'
                )
                for col in columns
            }
        ).drop(columns=columns)
    )

part_of_day_transformer = Pipeline(
    steps=[
        ('part_of_day_func', FunctionTransformer(func=part_of_day)),
        ('count_fre_encoder', CountFrequencyEncoder()),
        ('min_max_scaler', MinMaxScaler())
    ]
)

time_union_transformer = FeatureUnion(
    transformer_list=[
        ('time_transformer', time_transformer),
        ('part_of_day_transformer', part_of_day_transformer)
    ]
)

duration_log_transformer = FunctionTransformer(func=np.log)

# Define a custom function to determine if a flight is direct
def is_direct_flight(X):
    """
    Determine if a flight is direct.
    """
    return X.assign(
        is_direct_flight=X.total_stops.eq(0).astype(int)
    )

total_stops_transformer = FunctionTransformer(func=is_direct_flight)

# Combine all transformers into a column transformer
column_transformer = ColumnTransformer(transformers=[
    ('airline_transformer', airline_transformer, ['airline']),
    ('date_transformer', date_transformer, ['date_of_journey']),
    ('location_union_transformer', location_union_transformer, ['source', 'destination']),
    ('time_union_transformer', time_union_transformer, ['dep_time', 'arrival_time']),
    ('duration_log_transformer', duration_log_transformer, ['duration_minute']),
    ('total_stops_transformer', total_stops_transformer, ['total_stops'])
])

# Create the full pipeline including the RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('RF', RandomForestRegressor(max_depth=20, max_features=0.5, n_estimators=150))
])

def rf_train(X_train, y_train):
    """
    Train a RandomForestRegressor model using the provided training data and save the trained model.

    Parameters:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The target variable.

    Returns:
        None
    """
    try:
        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)
        
        # Save the trained model
        save(pipeline, 'RandomForestRegressor')
        print("Model trained and saved successfully.")
    
    except ValueError as e:
        print(f"Value error during training: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
