import pandas as pd
import numpy as np

S1 = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\S1_Passenger_Data_2019.csv', encoding='latin-1')
# drop rows from Stops 'Rissen' to ''Bahrenfeld'
S1 = S1.drop(S1.index[0:8])
# reverse the order of stops to calculate backward passengers onboard for each stop
S1 = S1.iloc[::-1].reset_index(drop=True)
S1['Passenger_Onboard_W'] = np.nan
S1['Passenger_Onboard_P'] = np.nan

# Splitting dataframe S1 into 2 dataframes: S1_P and S1_W becasue of different travel directions
S1_P = S1.loc[:, ['Stops', 'Embarking_P', 'Alighting_P', 'Passenger_Onboard_P']]
S1_P['Stops'] = S1_P['Stops'].apply(lambda x: str(x) + '_A')

S1_W = S1.loc[:, ['Stops' ,'Embarking_W', 'Alighting_W', 'Passenger_Onboard_W']]
S1_W['Stops'] = S1_W['Stops'].apply(lambda x: str(x) + '_B')

#############################################################################################################
# Direction toward Poppenbüttel: backwqrd calculation of passengers onboard for each stop
S1_P.loc[0, 'Passenger_Onboard_P'] = S1_P.loc[0, 'Alighting_P'].copy()
S1_P.loc[1, 'Passenger_Onboard_P'] = S1_P.loc[1, 'Alighting_P'].copy()
S1_P.loc[2, 'Passenger_Onboard_P'] = S1_P.loc[1, 'Alighting_P'].copy()
# Direction toward P, backward calculation start from last stop 'Poppenbüttel' alighting all the onboard passengers carry from 'Wellingsbüttel_A'
# calculate the number of passengers on board backwards for stops 'Wellingsbüttel_A' to 'Kornweg_A'
# passenger on board at next stop = passenger on board at previous stop - passenger alighting at next stop + passenger embarking at next stop
# this means: passenger on board at previous stop = passenger on board at next stop + passenger alighting at next stop - passenger embarking at next stop
# backward calculation: previous stop = i, next stop = i - 1
# Passenger_onboard_P[this stop] = Passenger_onboard_P[previous index stop] - Embarking_P[previous index stop] + Alighting_P[previous index stop]
def passenger_onboard_p(start, end):
    for i in range(start, end+1):
        S1_P.loc[i, 'Passenger_Onboard_P'] = S1_P.loc[i-1, 'Passenger_Onboard_P'] + S1_P.loc[i-1, 'Alighting_P'] - S1_P.loc[i-1, 'Embarking_P']
    return S1_P

passenger_onboard_p(3,4)

# train always splitting to tow parts at Ohlsdorf
# Get passenger Onboard for Kornweg direction train part before train arrival Kornweg
# passenger onboard at Kornweg = passenger onboard at Ohlsdorf - passenger alighting at Ohlsdorf + passenger embarking at Ohlsdorf
# passenger onboard leaving Ohlsdorf (Kronweg train part) = passenger onboard at Kornweg + passenger alighting at Kornweg - passenger embarking at Kornweg
train_part_kornweg = S1_P.loc[4, 'Passenger_Onboard_P'] + S1_P.loc[4, 'Alighting_P'] - S1_P.loc[4, 'Embarking_P']
# merg passengers for Ohlsdorf of train parts:  From Airport and from Kornweg
S1_P.loc[5, 'Passenger_Onboard_P'] = S1_P.loc[0, 'Passenger_Onboard_P'] + train_part_kornweg

passenger_onboard_p(6, 20)

S1_P
#############################################################################################################

# Direction toward Wedle: forward calculation of passengers onboard for each stop

S1_W.loc[0, 'Passenger_Onboard_W'] = S1_W.loc[0, 'Embarking_W'].copy()
S1_W.loc[1, 'Passenger_Onboard_W'] = S1_W.loc[1, 'Embarking_W'].copy()

# calculate the number of passengers on board for stops 'Wellingsbüttel_B' to 'Kornweg_B'
# Passenger_onboard_W = Passenger_onboard_W[previous index stop] + Embarking_P[this stop] - Disembarking_P[this stop]
def passenger_onboard_w(start, end):
    for i in range(start, end+1):
        S1_W.loc[i, 'Passenger_Onboard_W'] = S1_W.loc[i-1, 'Passenger_Onboard_W'] - S1_W.loc[i, 'Alighting_W'] + S1_W.loc[i, 'Embarking_W']
    return S1_W

passenger_onboard_w(2, 4)

# merg passengers for Ohlsdorf
S1_W.loc[5, 'Passenger_Onboard_W'] = S1_W.loc[0, 'Passenger_Onboard_W'] + S1_W.loc[4, 'Passenger_Onboard_W'] - S1_W.loc[5, 'Alighting_W'] + S1_W.loc[5, 'Embarking_W']
passenger_onboard_w(6, 20)

S1_W

# S1_to_P = S1_P.copy()
# S1_to_W = S1_W.copy()
# # change columne names for S1_P and S1_W before mergen
# S1_to_P.columns = ['Stops', 'Embarking', 'Alighting', 'Passenger_Onboard']
# S1_to_W.columns = ['Stops', 'Embarking', 'Alighting', 'Passenger_Onboard']
# #concate S1_to_P and S1_to_W
# S1_P_W = pd.concat([S1_to_P, S1_to_W], axis=0).reset_index(drop=True)
# S1_P_W.to_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Calculated_S1_Onboard_Passenger_Numbers.csv', index=False)

#############################################################################################################

# Get selected stops data for stops 'Ohlsdorf', 'Barmbek', 'Berliner Tor', 'Junfernstieg', 'Altona'
S1_P = S1_P.iloc[[5, 8, 13, 15, 20], :].reset_index(drop=True)
# reverse to correspond network direction
S1_P = S1_P.iloc[::-1].reset_index(drop=True)
S1_W = S1_W.iloc[[5, 8, 13, 15, 20], :].reset_index(drop=True)
S1_stops = pd.DataFrame()
S1_stops['Stops'] = pd.concat([S1_P['Stops'], S1_W['Stops']], axis=0).reset_index(drop=True)
S1_passenger_onboard = pd.DataFrame()
S1_passenger_onboard['Passenger_Onboard'] = pd.concat([S1_P['Passenger_Onboard_P'], S1_W['Passenger_Onboard_W']], axis=0).reset_index(drop=True) 
S1_passenger_demand = pd.concat([S1_stops, S1_passenger_onboard], axis=1).reset_index(drop=True)

##################################################################################
# Baseline changing rates:

# Get avg passenger on board this network: baseline 
baseline_onboard = S1_passenger_demand['Passenger_Onboard'].mean()
baseline_changing_rate = []
for index , passenger_data in S1_passenger_demand.iterrows():
    b_r = passenger_data['Passenger_Onboard'] / baseline_onboard -1
    baseline_changing_rate.append(b_r)

# handeling terminal stops, let passenger onboard capacity demand to be 0,
# terminal stops: 'Ohlsdorf A' and 'Altonn B' only disembarking passengers
baseline_changing_rate[4] = -1
baseline_changing_rate[-1] = -1 
S1_passenger_demand.loc[[4, 9],'Passenger_Onboard'] = 0

S1_passenger_demand['Changing_Rate_Baseline'] = baseline_changing_rate

#reduce passenger scale from 1260 mins (20 hours) t 10 mins
# for Passenger_Onboard_Ohlsdorf and Passenger_Onboard_Altona
S1_passenger_demand['Passenger_Onboard'] = S1_passenger_demand['Passenger_Onboard'] / 126
baseline_onboard = baseline_onboard / 126
# reduce train capacity from 500 to 50
# for Passenger_Onboard_P and Passenger_Onboard_W
S1_passenger_demand['Passenger_Onboard'] = S1_passenger_demand['Passenger_Onboard'] / 10
baseline_onboard = baseline_onboard / 10
baseline_changing_rate
#############################################################################################################

import train_timetable_S1_function as timetable
timetable_df = timetable.timetable_df
timetable_pivot = timetable.timetable_pivot
# Define simulation parameters
# Set random seed for reproducibility
np.random.seed(2023)
stops = list(range(10))
baseline_start = baseline_onboard
passenger_stop_std = 3.3333333333333335
# passenger_stop_std = 0
changing_rate = baseline_changing_rate
# if linear rate = 0, then baseline is constant over time
intensity_linear_rate = 0
# intensity_linear_rate = 3.5
#############################################################################################################
arrival_time = timetable_df['Arrival_Time'].copy()
bins = range(0, 10*60+1, 30)  # create bins of 30 minutes from 0 to 10*60 (total minutes in 10 hours)
labels = [f'{i/30}' for i in bins[:-1]]  # create labels for the bins in 0,1,2,3... format
group = pd.DataFrame()
group['Group'] = pd.cut(arrival_time, bins=bins, labels=labels, include_lowest=True)
arrival_time = pd.concat([arrival_time, group], axis=1)

# Now we know the order of baseline_onboard group in terms of time for each row
# baseline increasing linearly based on Group information in arrival_time
baseline = []
for index, data in arrival_time.iterrows():
    b_o = baseline_start + intensity_linear_rate*float(data['Group'])
    baseline.append(b_o)

baseline_series = pd.Series(baseline)
num_unique_values = baseline_series.nunique()

#############################################################################################################

# Define column names for train data
columns_train = ['Train_ID', 'Stop', 'Passenger_Demand', 'Passenger_Extra', 'Passenger_Onboard', 'STU_Onboard', 'Current_Load']

# Initialize DataFrame for train load data
init_load_data = pd.DataFrame(index = timetable_df.index, columns=columns_train)

# Populate DataFrame with initial data
init_load_data[['Train_ID', 'Stop']] = timetable_df[['Train_ID', 'Stop']]
init_load_data[['Passenger_Demand', 'Passenger_Extra', 'Passenger_Onboard', 'STU_Onboard', 'Current_Load']] = [np.nan, np.nan, np.nan, np.nan, np.nan]
init_load_data['Passenger_Onboard'] = np.nan
init_load_data['STU_Onboard'] = np.nan
init_load_data.loc[init_load_data['Train_ID'] == 0, 'Passenger_Extra'] = 0

# init_load_data = pd.concat([init_load_data, arrival_time], axis=1)
# init_load_data['Baseline'] = baseline
#############################################################################################################

def passenger_demand_generator(stops, baseline, changing_rate, passenger_stop_std, init_load_data):
    # Generate passenger demand for each stop base on baseline passenger demand and stop specific changing rate
    # iterate over init_load_data by index 
    # calculate passenger demand for each row base on same index in baseline, 
    # to determing the changing rate for the row, use the init_load_data['Stop'] integer number as stop_index to get the changing rate from changing_rate list
    # all passenger demand in init_load_data at stop are calculated by baseline * (1 + changing_rate[stop_index])
    for index, data in init_load_data.iterrows():
        stop_index = int(data['Stop'])
        if stop_index == 4 or stop_index == 9:
            init_load_data.loc[index, 'Passenger_Demand'] = 0
        else:
            init_load_data.loc[index, 'Passenger_Demand'] = np.maximum(0, np.round(np.random.normal(baseline[index] * (1 + changing_rate[stop_index]), passenger_stop_std), 0))
    return init_load_data

init_load_data = passenger_demand_generator(stops, baseline, changing_rate, passenger_stop_std, init_load_data)

init_load_data[init_load_data['Train_ID'] == 0] 
init_load_data[init_load_data['Stop'] == 1]

mu_list = []
for i in range(0, 10):
    mu = np.mean(init_load_data.loc[init_load_data['Stop'] == i, 'Passenger_Demand'])
    mu_list.append(mu)


