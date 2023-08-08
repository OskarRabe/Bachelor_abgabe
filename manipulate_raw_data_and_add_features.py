import glob
import os

import numpy as np
import pandas as pd
from meteostat import Point, Hourly
from tqdm import tqdm


def fetch_weather_data(df):
    # Get unique stations and their geographical points
    unique_stations = df[['PTCAR_ID', 'latitude', 'longitude']].drop_duplicates()

    # Get the time range and truncate to the nearest hour
    start_date = df['REAL_DATETIME_ARR'].dt.floor('H').min()
    end_date = df['REAL_DATETIME_ARR'].dt.floor('H').max()

    # Initialize DataFrame for weather data
    weather_data = pd.DataFrame()

    # Get weather data for each station
    for _, row in tqdm(unique_stations.iterrows(), total=unique_stations.shape[0], desc=f'Fetching weather data',
                       unit='station'):
        # Define geographical point
        location = Point(lat=row['latitude'], lon=row['longitude'])

        # Fetch hourly weather data
        data = Hourly(location, start_date, end_date)
        data = data.normalize()
        data = data.fetch()

        # Handle case where API returns no data
        if data.empty:
            print(f"No data returned for station: {row['PTCAR_ID']}")
            continue

        # Reset index
        data = data.reset_index()

        # Add PTCAR_ID column
        data['PTCAR_ID'] = row['PTCAR_ID']

        # Append to weather data DataFrame
        weather_data = pd.concat([weather_data, data], ignore_index=True)

    weather_data = weather_data.drop('tsun', axis=1)
    weather_data['snow'] = weather_data['snow'].fillna(value=0)
    weather_data['prcp'] = weather_data['prcp'].fillna(method='ffill')
    weather_data['coco'] = weather_data['coco'].fillna(method='ffill')
    weather_data['wpgt'] = weather_data['wpgt'].fillna(method='ffill')

    # Truncate REAL_DATETIME_ARR to the nearest hour
    df['REAL_DATETIME_ARR_trunc'] = df['REAL_DATETIME_ARR'].dt.floor('H')
    df['REAL_DATETIME_DEP_trunc'] = df['REAL_DATETIME_DEP'].dt.floor('H')
    df['PLANNED_DATETIME_ARR_NEXT_trunc'] = df['PLANNED_DATETIME_ARR_NEXT'].dt.floor('H')

    # Create a new column by combining REAL_DATETIME_ARR_trunc and REAL_DATE_DEP_trunc
    df['MERGE_DATETIME'] = df['REAL_DATETIME_ARR_trunc'].combine_first(df['REAL_DATETIME_DEP_trunc'])

    # Merge weather data with the original DataFrame on PTCAR_ID and truncated datetime
    df = df.merge(weather_data, left_on=['PTCAR_ID', 'MERGE_DATETIME'], right_on=['PTCAR_ID', 'time'], how='left')
    df = df.merge(weather_data, left_on=['next_stop_encoded', 'PLANNED_DATETIME_ARR_NEXT_trunc'],
                  right_on=['PTCAR_ID', 'time'], how='left', suffixes=('', '_next'))

    # Drop the temporary columns and columns that are not needed
    df = df.drop(
        ['MERGE_DATETIME', 'REAL_DATETIME_ARR_trunc', 'REAL_DATETIME_DEP_trunc', 'PLANNED_DATETIME_ARR_NEXT_trunc',
         'time', 'time_next', 'PTCAR_ID_next'], axis=1)

    return df


def remove_invalid_routes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with invalid routes from the input DataFrame.
    - Rows with missing latitude or longitude values
    - Rows with unique_journey_id where the max of the stop_number column is less than 6
    - Rows with planned_time_to_next_stop less than 0, ignoring NaN values
    """
    # Check for invalid stops (missing latitude or longitude values)
    invalid_stops = dataframe[['latitude', 'longitude']].isna().any(axis=1)

    # Get unique invalid route IDs
    invalid_routes = dataframe[invalid_stops]['unique_journey_id'].unique()

    # Create a DataFrame grouping by unique_journey_id
    group_df = dataframe.groupby('unique_journey_id').agg(max_stop_number=('stop_number', 'max'),
        min_planned_time_to_next_stop=('planned_time_to_next_stop', 'min'))

    # Identify routes where the max of the stop_number column is less than 6
    invalid_routes = np.concatenate([invalid_routes, group_df[group_df['max_stop_number'] < 6].index.values])

    # Identify routes where planned_time_to_next_stop is less than 0 (ignoring NaN values)
    invalid_routes = np.concatenate(
        [invalid_routes, group_df[group_df['min_planned_time_to_next_stop'] < 0].index.values])

    # Filter out rows with invalid route IDs
    valid_df = dataframe[~dataframe['unique_journey_id'].isin(invalid_routes)]

    return valid_df


# Function to create a unique journey ID based on the sequence of stops using 'PTCAR_NO'
def create_unique_journey_id(group):
    global unique_journey_id_counter
    # Creating a sequence representing the train's stops using 'PTCAR_NO'
    sequence = tuple(group['PTCAR_ID'].tolist())

    # Checking if the sequence already has a unique journey ID assigned
    unique_journey_id = unique_journey_sequences.get(sequence)
    if unique_journey_id is None:
        # Creating a new unique journey ID and assigning it to the sequence
        unique_journey_id = unique_journey_id_counter
        unique_journey_sequences[sequence] = unique_journey_id
        unique_journey_id_counter += 1

    # Assigning the unique journey ID to all rows in the group
    group['unique_journey_id'] = unique_journey_id
    return group


def add_station_data(station_data: pd.DataFrame, punctuality_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Add information about the current station and the next station to a DataFrame of train trips.
    '''

    # List of columns to be added to the original DataFrame
    columns_to_add = ['Classification_EN', 'Complete_name_in_Dutch', 'latitude', 'longitude', ]

    # Create a mapping dictionary for each column
    column_mapping = {column: station_data.set_index(keys='PTCAR_ID')[column].to_dict() for column in columns_to_add}

    # Use the mapping dictionaries to add the information for each column for the current and next stations
    for column, mapping in column_mapping.items():
        punctuality_data[column] = punctuality_data['PTCAR_ID'].map(mapping)
        punctuality_data[f'next_{column}'] = punctuality_data['next_stop_encoded'].map(mapping)
    punctuality_data.reset_index(drop=True, inplace=True)
    return punctuality_data


# doesnt work because dataset doesnt include 90% of the stations
def add_monthly_punctuality_data(raw_punctuality_df: pd.DataFrame,
                                 monthly_punctuality_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add monthly punctuality data to a DataFrame of train trips.
    '''
    merged_df = pd.merge(raw_punctuality_df, monthly_punctuality_df, left_on=['PTCAR_ID', 'month', 'year'],
                         right_on=['Operational point ID', 'Date_month', 'Date_year'], how='left')

    # Select the desired columns
    merged_df = merged_df.drop(columns=['Operational point ID', 'Date', 'Date_month', 'Date_year'])

    return merged_df


def process_parquet_file(file_path: str) -> None:
    df = pd.read_parquet(file_path)
    df = df.drop(columns=['THOP1_COD', 'CIRC_TYP'])
    # Create new columns with the encoded values
    df['RELATION_encoded'] = pd.Categorical(df['RELATION']).codes
    df['TRAIN_SERV_encoded'] = pd.Categorical(df['TRAIN_SERV']).codes  # eher onehot weil nur 2 werte
    df['RELATION_DIRECTION_encoded'] = pd.Categorical(df['RELATION_DIRECTION']).codes

    # Convert DATDEP to datetime
    df['DATDEP'] = pd.to_datetime(df['DATDEP'], format='%d%b%Y')
    # Extract the day of the week
    df['day_of_week'] = df['DATDEP'].dt.day_name()
    # Extract day, month, year
    df['day'] = df['DATDEP'].dt.day
    df['month'] = df['DATDEP'].dt.month
    df['year'] = df['DATDEP'].dt.year

    # Create copies of the time columns before converting them to minutes
    for col in ['REAL_TIME_DEP', 'REAL_TIME_ARR', 'PLANNED_TIME_DEP', 'PLANNED_TIME_ARR']:
        df[col + '_original'] = df[col]

    # Convert the time columns to datetime, then extract the total minutes since the day started
    for col in ['REAL_TIME_DEP', 'REAL_TIME_ARR', 'PLANNED_TIME_DEP', 'PLANNED_TIME_ARR']:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S')
        df[col] = df[col].dt.hour * 60 + df[col].dt.minute
    df = df.rename(columns={'PTCAR_NO': 'PTCAR_ID'})
    df['journey_id'] = df.groupby(['DATDEP', 'TRAIN_NO']).ngroup()
    # Create a column with the stop number for each stop in a journey
    df['stop_number'] = df.groupby('journey_id').cumcount()
    # Create a column with the number of remaining stops for each stop in a journey
    total_stops = df.groupby('journey_id')['stop_number'].transform('max')
    df['remaining_stops'] = total_stops - df['stop_number']
    # df.info(verbose=True, show_counts=True)

    df = df.groupby(['journey_id', 'DATDEP']).apply(create_unique_journey_id)

    # Group the data by journey_id and create a column with the journey route
    df = df.reset_index(drop=True)
    df['journey_route_encoded'] = (
        df.sort_values(by=['journey_id', 'stop_number']).groupby('journey_id')['PTCAR_ID'].transform(
            lambda x: '->'.join(x.astype(str))))

    df['journey_route'] = (
        df.sort_values(by=['journey_id', 'stop_number']).groupby('journey_id')['PTCAR_LG_NM_NL'].transform(
            lambda x: '->'.join(x.astype(str))))
    # Sort the dataframe by 'journey_id' and 'stop_number' to ensure stops are in correct order
    df = df.sort_values(['journey_id', 'stop_number'])
    # Shift the 'PTCAR_ID' column up by one row within each 'journey_id' group
    df['next_stop_encoded'] = df.groupby('journey_id')['PTCAR_ID'].shift(-1)
    # Shift the 'PTCAR_ID' column up by one row within each 'journey_id' group
    df['next_stop'] = df.groupby('journey_id')['PTCAR_LG_NM_NL'].shift(-1)

    # Convert the date and time columns to datetime format and combine them
    df['REAL_DATETIME_ARR'] = pd.to_datetime(df['REAL_DATE_ARR'] + ' ' + df['REAL_TIME_ARR_original'],
                                             format='%d%b%Y %H:%M:%S')
    df['REAL_DATETIME_DEP'] = pd.to_datetime(df['REAL_DATE_DEP'] + ' ' + df['REAL_TIME_DEP_original'],
                                             format='%d%b%Y %H:%M:%S')
    df['PLANNED_DATETIME_ARR'] = pd.to_datetime(df['PLANNED_DATE_ARR'] + ' ' + df['PLANNED_TIME_ARR_original'],
                                                format='%d%b%Y %H:%M:%S')
    df['PLANNED_DATETIME_DEP'] = pd.to_datetime(df['PLANNED_DATE_DEP'] + ' ' + df['PLANNED_TIME_DEP_original'],
                                                format='%d%b%Y %H:%M:%S')

    # Shift the 'PLANNED_DATE_ARR' and 'PLANNED_TIME_ARR' columns up by one row within each 'journey_id' group
    df['PLANNED_DATE_ARR_NEXT'] = df.groupby('journey_id')['PLANNED_DATE_ARR'].shift(-1)
    df['PLANNED_TIME_ARR_NEXT'] = df.groupby('journey_id')['PLANNED_TIME_ARR_original'].shift(-1)
    # Combine the shifted columns to create the planned arrival datetime for the next station
    df['PLANNED_DATETIME_ARR_NEXT'] = pd.to_datetime(df['PLANNED_DATE_ARR_NEXT'] + ' ' + df['PLANNED_TIME_ARR_NEXT'],
                                                     format='%d%b%Y %H:%M:%S')
    # Remove the temporary columns
    df = df.drop(columns=['REAL_TIME_DEP_original', 'REAL_TIME_ARR_original', 'PLANNED_TIME_DEP_original',
                          'PLANNED_TIME_ARR_original', 'PLANNED_DATE_ARR_NEXT', 'PLANNED_TIME_ARR_NEXT'])
    df = add_station_data(station_data=op_df, punctuality_data=df)
    # add the planned time to the next stop as a column
    df['planned_time_to_next_stop'] = (df['PLANNED_DATETIME_ARR_NEXT'] - df[
        'PLANNED_DATETIME_DEP']).dt.total_seconds() / 60
    df = remove_invalid_routes(dataframe=df)
    df = df.convert_dtypes()
    df = fetch_weather_data(df)
    # df = add_monthly_punctuality_data(df, monthly_punctuality)
    df['median_delay_next_station'] = df['next_stop_encoded'].map(
        df.groupby('PTCAR_ID')['DELAY_ARR'].median().fillna(0))
    df = df.convert_dtypes()
    # Downcast integer columns
    icols = df.select_dtypes('integer').columns
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    output_file_name = file_path.replace('Raw', 'Processed').replace('Data_raw', 'Data_processed')
    df = df.sort_values(by=['journey_id', 'stop_number'])
    df.to_parquet(output_file_name)


op_df = pd.read_parquet(path='Data/Processed/Network/Operational_point_of_the_railway_network.parquet')
# Get a list of all parquet files in the directory
files = glob.glob('Data/Raw/Punctuality/*.parquet')

unique_journey_sequences = {}
unique_journey_id_counter = 0
# monthly_punctuality = pd.read_csv('Data/Processed/Punctuality/Monthly_punctuality_by_stopping_point.csv')
# monthly_punctuality['Date'] = pd.to_datetime(monthly_punctuality['Date'], format='%Y-%m')
# monthly_punctuality['Date_month'] = monthly_punctuality['Date'].dt.month
# monthly_punctuality['Date_year'] = monthly_punctuality['Date'].dt.year

# Iterate through the files and process each one
for file_path in tqdm(files, unit='file'):
    file_name = os.path.basename(file_path)
    tqdm.write(f'Processing {file_name}')
    process_parquet_file(file_path)

# Janur und Dezember haben am wenigsten fehlende daten
