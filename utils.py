import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fetch_race_results(years):
    races = []
    for year in years:
        url = f"https://ergast.com/api/f1/{year}/results.json?limit=1000"
        response = requests.get(url)
        data = response.json()
        
        if 'MRData' in data and 'RaceTable' in data['MRData'] and 'Races' in data['MRData']['RaceTable']:
            races_data = data['MRData']['RaceTable']['Races']
            for race in races_data:
                circuit_name = race['Circuit']['circuitName']
                
                if 'Results' in race:
                    race_results = race['Results']
                    for result in race_results:
                        race_info = {
                            'Year': year,
                            'Circuit': circuit_name,
                            'Driver': result['Driver']['driverId'],
                            'Constructor': result['Constructor']['constructorId'],
                            'Grid': int(result['grid']),
                            'Position': int(result['position']),
                            'FastestLapTime': result['FastestLap']['Time']['time'] if 'FastestLap' in result else None,
                            'Status': result['status']
                        }
                        races.append(race_info)
        else:
            print(f"No race data found for year {year}")

    return races

def preprocess_data(df):
    label_encoders = {}

    categorical_cols = ['Driver', 'Constructor', 'Circuit']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

def calculate_grid_position_trends(df):
    df['Position_Change'] = df['Grid'] - df['Position']
    driver_position_trend = df.groupby('Driver')['Position_Change'].mean().reset_index()
    driver_position_trend.columns = ['Driver', 'Avg_Position_Change']
    df = df.merge(driver_position_trend, on='Driver', how='left')
    return df

def calculate_driver_constructor_synergy(df):
    df['Position_Change'] = df['Grid'] - df['Position']
    synergy = df.groupby(['Driver', 'Constructor'])['Position_Change'].mean().reset_index()
    synergy.columns = ['Driver', 'Constructor', 'Driver_Constructor_Synergy']
    df = df.merge(synergy, on=['Driver', 'Constructor'], how='left')
    return df

def calculate_consistency_metrics(df):
    driver_std_dev = df.groupby('Driver')['Position'].std().reset_index()
    driver_std_dev.columns = ['Driver', 'Position_StdDev']
    df = df.merge(driver_std_dev, on='Driver', how='left')
    return df

def calculate_race_incident_rates(df):
    df['Incident'] = df['Status'].apply(lambda x: 0 if x == 'Finished' else 1)
    incident_rates = df.groupby('Driver')['Incident'].mean().reset_index()
    incident_rates.columns = ['Driver', 'Incident_Rate']
    df = df.merge(incident_rates, on='Driver', how='left')
    return df

def calculate_circuit_performance(df):
    circuit_performance = df.groupby(['Driver', 'Circuit'])['Position'].mean().reset_index()
    circuit_performance.columns = ['Driver', 'Circuit', 'Circuit_Performance']
    df = df.merge(circuit_performance, on=['Driver', 'Circuit'], how='left')
    return df

def calculate_qualifying_performance(df):
    qualifying_performance = df.groupby('Driver')['Grid'].mean().reset_index()
    qualifying_performance.columns = ['Driver', 'Avg_Qualifying_Position']
    df = df.merge(qualifying_performance, on='Driver', how='left')
    return df
