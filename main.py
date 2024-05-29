import pandas as pd
from src.scripts.data_cleaning import clean_df
from src.scripts.model_training import rf_train
from src.scripts.model_evaluation import evaluation
from src.scripts.model_prediction import prediction
from src.utils.load_file import load
from src.utils.split_dataset import split

MAIN_DATASET_FILE_NAME = 'flight_price.csv'
MODEL_NAME = 'RandomForestRegressor.pkl'

def convert_to_time(X_train, X_val, X_test):
    X_train['dep_time'] = pd.to_datetime(X_train['dep_time'].astype(str), format='%H:%M:%S')
    X_train['arrival_time'] = pd.to_datetime(X_train['arrival_time'].astype(str), format='%H:%M:%S')

    X_val['dep_time'] = pd.to_datetime(X_val['dep_time'].astype(str), format='%H:%M:%S')
    X_val['arrival_time'] = pd.to_datetime(X_val['arrival_time'].astype(str), format='%H:%M:%S')

    X_test['dep_time'] = pd.to_datetime(X_test['dep_time'].astype(str), format='%H:%M:%S')
    X_test['arrival_time'] = pd.to_datetime(X_test['arrival_time'].astype(str), format='%H:%M:%S')
    

def train():  
    df = load(file_name=MAIN_DATASET_FILE_NAME, file_type='csv')
    df = clean_df(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split(df)
    convert_to_time(X_train, X_val, X_test)
    rf_train(X_train, y_train)
    model = load(file_name=MODEL_NAME, file_type='pkl')
    
    print('================================')
    evaluation(model, 'train', X_train, y_train)
    print('================================')
    evaluation(model, 'vaidation', X_val, y_val)
    print('================================')
    evaluation(model, 'test', X_test, y_test)
    print('================================')
    

def test():
    query = pd.DataFrame(
        {
            'airline': ['Multiple Carriers'],
            'date_of_journey': ['2019-05-21'],
            'source': ['Delhi'],
            'destination': ['Cochin'],
            'dep_time': ['02:15:00'],
            'arrival_time': ['11:50:00'],
            'duration_minute': [575],
            'total_stops': [1]
        }
    )
    print(prediction(query))
    
def main():
    train()
    test()

if __name__ == '__main__':
    main()