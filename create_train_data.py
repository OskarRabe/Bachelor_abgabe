import os
import pandas as pd
import glob
from tqdm import tqdm
# https://dev.meteostat.net/formats.html#meteorological-data-units =weather condition codes
# files = ['Data/Processed/Punctuality/Data_processed_punctuality_202201.parquet']
files = glob.glob(pathname='Data/Processed/Punctuality/*.parquet')


def process_time_columns(data):
    # Replace NaN in 'real_time_arrival_minutes' with corresponding 'real_time_departure_minutes'
    data['real_time_arrival_minutes'].fillna(data['real_time_departure_minutes'], inplace=True)

    # Replace NaN in 'real_time_departure_minutes' with corresponding 'real_time_arrival_minutes'
    data['real_time_departure_minutes'].fillna(data['real_time_arrival_minutes'], inplace=True)

    # Convert 'real_time_arrival_minutes' and 'real_time_departure_minutes' to integer
    data['real_time_arrival_minutes'] = data['real_time_arrival_minutes'].astype(int)
    data['real_time_departure_minutes'] = data['real_time_departure_minutes'].astype(int)

    # Convert 'real_time_arrival_minutes' and 'real_time_departure_minutes' to hour and minute
    data['arrival_hour'] = data['real_time_arrival_minutes'] // 60
    data['arrival_minute'] = data['real_time_arrival_minutes'] % 60
    data['departure_hour'] = data['real_time_departure_minutes'] // 60
    data['departure_minute'] = data['real_time_departure_minutes'] % 60
    data['arrival_delay_seconds'].fillna(0, inplace=True)
    # Drop the original columns
    data = data.drop(columns=['real_time_arrival_minutes', 'real_time_departure_minutes'])
    data.fillna(-1, inplace=True)

    return data



def process_files(files):
    """
    This function reads parquet files from a directory, encodes certain columns, and saves the resulting dataframes as parquet files in a different directory.
    """
    # Iterate through the files and process each one
    for file_path in tqdm(iterable=files, unit='file'):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        tqdm.write(f'Processing {file_name}')
        df = pd.read_parquet(file_path)
        # Create new columns with the encoded values
        df['LINE_NO_DEP_encoded'] = pd.Categorical(values=df['LINE_NO_DEP']).codes
        df['LINE_NO_ARR_encoded'] = pd.Categorical(values=df['LINE_NO_ARR']).codes
        df['Classification_EN_encoded'] = pd.Categorical(values=df['Classification_EN']).codes
        df['next_Classification_EN_encoded'] = pd.Categorical(values=df['next_Classification_EN']).codes
        df['day_of_week_encoded'] = pd.Categorical(values=df['day_of_week']).codes
        df = df.drop(columns=['LINE_NO_DEP', 'LINE_NO_ARR', 'Classification_EN', 'next_Classification_EN', 'RELATION',
                              'TRAIN_SERV', 'RELATION_DIRECTION', 'PTCAR_LG_NM_NL', 'PLANNED_DATE_ARR',
                              'PLANNED_DATE_DEP', 'REAL_DATE_ARR', 'REAL_DATE_DEP', 'PLANNED_TIME_ARR',
                              'PLANNED_TIME_DEP', 'journey_route', 'journey_route_encoded', 'next_stop',
                              'Complete_name_in_Dutch', 'next_Complete_name_in_Dutch', 'day_of_week',
                              'REAL_DATETIME_DEP', 'PLANNED_DATETIME_ARR', 'PLANNED_DATETIME_DEP',
                              'PLANNED_DATETIME_ARR_NEXT', 'DATDEP', 'REAL_DATETIME_ARR'])
        column_names = {'TRAIN_NO': 'train_number', 'PTCAR_ID': 'current_station_id',
                        'REAL_TIME_ARR': 'real_time_arrival_minutes', 'REAL_TIME_DEP': 'real_time_departure_minutes',
                        'DELAY_ARR': 'arrival_delay_seconds', 'DELAY_DEP': 'departure_delay_seconds',
                        'RELATION_encoded': 'train_service_number', 'TRAIN_SERV_encoded': 'train_operator',
                        'RELATION_DIRECTION_encoded': 'relation_direction', 'day': 'day_of_month', 'month': 'month', 'year': 'year',
                        'journey_id': 'journey_id', 'stop_number': 'current_stop_number', 'remaining_stops': 'remaining_stops',
                        'unique_journey_id': 'unique_journey_id', 'next_stop_encoded': 'next_station_id',
                        'latitude': 'current_station_latitude', 'next_latitude': 'next_station_latitude',
                        'longitude': 'current_station_longitude', 'next_longitude': 'next_station_longitude', 'temp': 'temperature',
                        'dwpt': 'dew_point', 'rhum': 'relative_humidity', 'prcp': 'precipitation', 'snow': 'snow',
                        'wdir': 'wind_direction', 'wspd': 'wind_speed', 'wpgt': 'wind_peak_gust', 'pres': 'pressure',
                        'coco': 'weather_condition_code', 'temp_next': 'next_station_temperature',
                        'dwpt_next': 'next_station_dew_point', 'rhum_next': 'next_station_relative_humidity',
                        'prcp_next': 'next_station_precipitation', 'snow_next': 'next_station_snow',
                        'wdir_next': 'next_station_wind_direction', 'wspd_next': 'next_station_wind_speed',
                        'wpgt_next': 'next_station_wind_peak_gust', 'pres_next': 'next_station_pressure',
                        'coco_next': 'next_station_weather_condition_code',
                        'median_delay_next_station': 'median_delay_next_station', 'LINE_NO_DEP_encoded': 'departure_line_encoded',
                        'LINE_NO_ARR_encoded': 'arrival_line_encoded',
                        'Classification_EN_encoded': 'current_classification_encoded',
                        'next_Classification_EN_encoded': 'next_classification_encoded',
                        'day_of_week_encoded': 'day_of_week_encoded'}
        # Rename the columns using the mapping
        df = df.rename(columns=column_names)
        df = df.drop(columns=['next_station_pressure','pressure','dew_point','relative_humidity','precipitation','snow',
                              'wind_direction','wind_speed',
                              'next_station_dew_point','next_station_relative_humidity',
                              'next_station_precipitation','next_station_snow','next_station_wind_direction',
                              'next_station_wind_speed','year','train_operator'])
        df = process_time_columns(df)
        df.info(verbose=True, show_counts=True)
        df.to_csv(f'Data/Processed/Train/{file_name}.csv', index=False)


process_files(files=files)
df =pd.read_csv('Data/Processed/Train/Data_processed_punctuality_202201.csv')
# df =pd.read_parquet('Data/Processed/Punctuality/Data_processed_punctuality_202201.parquet')

# # df = pd.read_parquet('Data/Processed/Punctuality/Data_processed_punctuality_202201.parquet')
df.info(verbose=True, show_counts=True)

df.head(50_000).to_csv('50k_Data_processed_punctuality_202201.csv', index=False)
# df['LINE_NO_DEP_encoded'] = pd.Categorical(values=df['LINE_NO_DEP']).codes
# df['LINE_NO_ARR_encoded '] = pd.Categorical(values=df['LINE_NO_ARR']).codes
# df['Classification_EN_encoded '] = pd.Categorical(values=df['Classification_EN']).codes
# df['next_Classification_EN_encoded '] = pd.Categorical(values=df['next_Classification_EN']).codes
# df['day_of_week_encoded'] = pd.Categorical(values=df['day_of_week']).codes
# df = df.drop(
#     columns=['LINE_NO_DEP', 'LINE_NO_ARR', 'Classification_EN', 'next_Classification_EN', 'RELATION', 'TRAIN_SERV',
#              'RELATION_DIRECTION', 'PTCAR_LG_NM_NL', 'PLANNED_DATE_ARR', 'PLANNED_DATE_DEP', 'REAL_DATE_ARR',
#              'REAL_DATE_DEP', 'PLANNED_TIME_ARR', 'PLANNED_TIME_DEP', 'journey_route', 'journey_route_encoded',
#              'next_stop', 'Complete_name_in_Dutch', 'next_Complete_name_in_Dutch', 'day_of_week', 'REAL_DATETIME_DEP',
#              'PLANNED_DATETIME_ARR', 'PLANNED_DATETIME_DEP', 'PLANNED_DATETIME_ARR_NEXT', 'DATDEP',
#              'REAL_DATETIME_ARR'])
# df.info(verbose=True, show_counts=True)
# icols = df.select_dtypes('integer').columns
# df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
# df.info(verbose=True, show_counts=True)
