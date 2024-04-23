import pandas as pd
import numpy as np

S1 = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\S1_Passenger_Data_2019.csv', encoding='latin-1')
# drop rows from Stops 'Rissen' to ''Bahrenfeld'
S1 = S1.drop(S1.index[0:8])
# reverse rows last row becons first row and first row becons last row, reset index
S1 = S1.iloc[::-1].reset_index(drop=True)
S1['Passenger_Onboard_W'] = np.nan
S1['Passenger_Onboard_P'] = np.nan

# Splitting dataframe S1 into 2 dataframes: S1_P and S1_W interms of columns
S1_P = S1.loc[:, ['Stops', 'Embarking_P', 'Alighting_P', 'Passenger_Onboard_P']]
S1_W = S1.loc[:, ['Stops' ,'Embarking_W', 'Alighting_W', 'Passenger_Onboard_W']]


S1_P.loc[0, 'Passenger_Onboard_P'] = S1_P.loc[0, 'Alighting_P'].copy()
S1_P.loc[1, 'Passenger_Onboard_P'] = S1_P.loc[1, 'Alighting_P'].copy()
S1_P.loc[2, 'Passenger_Onboard_P'] = S1_P.loc[1, 'Alighting_P'].copy()

S1_W.loc[0, 'Passenger_Onboard_W'] = S1_W.loc[0, 'Embarking_W'].copy()
S1_W.loc[1, 'Passenger_Onboard_W'] = S1_W.loc[1, 'Embarking_W'].copy()

# Directing W
# calculate the number of passengers on board for stops 'Wellingsbüttel' to 'Kornweg'
# Passenger_onboard_W = Passenger_onboard_W[previous index stop] + Embarking_P[this stop] - Disembarking_P[this stop]
def passenger_onboard_w(start, end):
    for i in range(start, end+1):
        S1_W.loc[i, 'Passenger_Onboard_W'] = S1_W.loc[i-1, 'Passenger_Onboard_W'] - S1_W.loc[i, 'Alighting_W'] + S1_W.loc[i, 'Embarking_W']
    return S1_W

passenger_onboard_w(2, 4)

# merg passengers for Ohlsdorf
S1_W.loc[5, 'Passenger_Onboard_W'] = S1_W.loc[0, 'Passenger_Onboard_W'] + S1_W.loc[4, 'Passenger_Onboard_W'] - S1_W.loc[5, 'Alighting_W'] + S1_W.loc[5, 'Embarking_W']
passenger_onboard_w(6, 20)

# Directing P, backward calculation from last stop 'Poppenbüttel' alighting all at 'Wellingsbüttel' loaded passengers
# calculate the number of passengers on board for stops 'Wellingsbüttel' to 'Kornweg'
# passenger on board at next stop = passenger on board at previous stop - passenger alighting at next stop + passenger embarking at next stop
# this means: passenger on board at previous stop = passenger on board at next stop + passenger alighting at next stop - passenger embarking at next stop
# backward calculation: previous stop = this stop, next stop = previous index stop
# Passenger_onboard_P[this stop] = Passenger_onboard_P[previous index stop] - Embarking_P[previous index stop] + Alighting_P[previous index stop]
def passenger_onboard_p(start, end):
    for i in range(start, end+1):
        S1_P.loc[i, 'Passenger_Onboard_P'] = S1_P.loc[i-1, 'Passenger_Onboard_P'] - S1_P.loc[i-1, 'Embarking_P'] + S1_P.loc[i-1, 'Alighting_P']
    return S1_P

passenger_onboard_p(3,4)

# Get passenger Onboard for kornweg train part before train arrival kornweg
train_part_kornweg = S1_P.loc[4, 'Passenger_Onboard_P'] - S1_P.loc[4, 'Embarking_P'] + S1_P.loc[4, 'Alighting_P']
# merg passengers for Ohlsdorf of train parts: Airport and Kornweg
S1_P.loc[5, 'Passenger_Onboard_P'] = S1_P.loc[0, 'Passenger_Onboard_P'] + train_part_kornweg

passenger_onboard_p(6, 20)


# calculate percentage of Embarking_P and Embarking_W based on the total number of Embarking_P + Embarking_W
embarking_p = S1['Embarking_P'] / (S1['Embarking_P'] + S1['Embarking_W'])
embarking_w = S1['Embarking_W'] / (S1['Embarking_P'] + S1['Embarking_W'])

S1_P['Embarking_P_percentage'] = embarking_p
S1_W['Embarking_W_percentage'] = embarking_w

# new dataframe with columns: 'Stops', 'Embarking_P_percentage', 'Embarking_W_percentage','Passenger_Onboard_P', 'Passenger_Onboard_W'
S1_passenger = pd.concat([S1_P['Stops'], S1_P['Passenger_Onboard_P'], S1_W['Passenger_Onboard_W']], axis=1)

#Get spliting percentage
station_splitting_percentage = pd.concat([S1_P['Stops'], S1_P['Embarking_P_percentage'], S1_W['Embarking_W_percentage']], axis=1)

####################################################################################################################

# Get selected stops data for stops 'Ohlsdorf', 'Barmbek', 'Berliner Tor', 'Junfernstieg', 'Altona'
S1_passenger_demand = S1_passenger.iloc[[5, 8, 13, 15, 20], :].reset_index(drop=True)
S1_passenger_demand = S1_passenger_demand.rename(columns={'Passenger_Onboard_W': 'Passenger_Onboard_A'})
S1_passenger_demand = S1_passenger_demand.rename(columns={'Passenger_Onboard_P': 'Passenger_Onboard_O'})


S1_passenger_demand = S1_passenger_demand.reindex(columns=['Stops', 'Passenger_Onboard_O', 'Changing_Rate_O',  'Passenger_Onboard_A','Changing_Rate_A'])
# revers the direction to correspond network direction
S1_passenger_demand = S1_passenger_demand.iloc[::-1].reset_index(drop=True)


#########################################################################################################################
# To prev_stop changing rate: another idea to get changing rates

# Adjust Network to circular network in this work
S1_passenger_demand.loc[0,'Changing_Rate_O'] = 0
S1_passenger_demand.loc[4,'Changing_Rate_A'] = 0
S1_passenger_demand.loc[4,'Passenger_Onboard_O'] = 0
S1_passenger_demand.loc[0,'Passenger_Onboard_A'] = 0

# Direction toward Ohlsdorf:
def changing_rate_O(start, end):
    for i in range(start, end+1):
        S1_passenger_demand.loc[i,'Changing_Rate_O'] = (S1_passenger_demand.loc[i,'Passenger_Onboard_O'] / S1_passenger_demand.loc[i-1,'Passenger_Onboard_O']) - 1
    return S1_passenger_demand

changing_rate_O(1,4)

# Direction toward Altona:
# changing_rate = (Passenger_Onboard_W[this stop]) / (Passenger_Onboard_W[previous stop]) - 1
def changing_rate_A(start, end):
    for i in range(start, end+1):
        S1_passenger_demand.loc[i,'Changing_Rate_A'] = (S1_passenger_demand.loc[i,'Passenger_Onboard_A'] / S1_passenger_demand.loc[i+1,'Passenger_Onboard_A']) - 1
    return S1_passenger_demand

changing_rate_A(0,3)

# devide changing rate from real word data
#reduce passenger scale from 1260 mins (20 hours) t 10 mins
# for Passenger_Onboard_Ohlsdorf and Passenger_Onboard_Altona
S1_passenger_demand['Passenger_Onboard_A'] = S1_passenger_demand['Passenger_Onboard_A'] / 126
S1_passenger_demand['Passenger_Onboard_O'] = S1_passenger_demand['Passenger_Onboard_O'] / 126

# reduce train capacity from 500 to 50
# for Passenger_Onboard_P and Passenger_Onboard_W
S1_passenger_demand['Passenger_Onboard_A'] = S1_passenger_demand['Passenger_Onboard_A'] / 10
S1_passenger_demand['Passenger_Onboard_O'] = S1_passenger_demand['Passenger_Onboard_O'] / 10

# initial passenger demand for Altona with direction toward Ohlsdorf
passenger_altona_avg = S1_passenger_demand.loc[0, 'Passenger_Onboard_O'].astype(float)
# initial passenger demand for Ohlsdorf with direction toward Altona
passenger_ohlsdorf_avg = S1_passenger_demand.loc[4, 'Passenger_Onboard_A'].astype(float)


changing_rate_O = S1_passenger_demand['Changing_Rate_O']
changing_rate_A = S1_passenger_demand['Changing_Rate_A']
# reverse rows of direction toward Altona to create  circular network
changing_rate_A = changing_rate_A.iloc[::-1].reset_index(drop=True)
# turn changing_rate to list
changing_rate_O = changing_rate_O.tolist()
changing_rate_A = changing_rate_A.tolist()
changing_rates = changing_rate_O + changing_rate_A

#########################################################################################################################

import train_timetable_S1_function as timetable
timetable_df = timetable.timetable_df
timetable_pivot = timetable.timetable_pivot
# Define simulation parameters
# Set random seed for reproducibility
np.random.seed(2023)
stops = list(range(10))
passenger_demand_0_5 = [None, None]
# passenger_demand_std = 0
passenger_demand_std = 3.3333333333333335
intensity_linear_rate = 0 # passenger intensity constant over time
# intensity_linear_rate = 3.5

# Define column names for train data
columns_train = ['Train_ID', 'Stop', 'Passenger_Demand', 'Passenger_Extra', 'Passenger_Onboard', 'STU_Onboard', 'Current_Load']

# Initialize DataFrame for train load data
init_load_data = pd.DataFrame(index = timetable_df.index, columns=columns_train)

# Populate DataFrame with initial data
init_load_data[['Train_ID', 'Stop']] = timetable_df[['Train_ID', 'Stop']]
init_load_data[['Passenger_Demand', 'Passenger_Extra', 'Passenger_Onboard', 'STU_Onboard', 'Current_Load']] = [np.nan, np.nan, np.nan, np.nan, np.nan]
# Train status: 0 = not sarted, 1 = running, 2 = standing, 3 = terminated
init_load_data['Passenger_Onboard'] = np.nan
init_load_data['STU_Onboard'] = np.nan
init_load_data.loc[init_load_data['Train_ID'] == 0, 'Passenger_Extra'] = 0

#########################################################################################################################

def generate_start_passenger_onboard(timetable_df, passenger_altona_avg, passenger_ohlsdorf_avg, intensity_linear_rate, passenger_demand_std):
    # find the columns where Stop equal 0 or equal 5
    # only sart stops need to time dependent passenger demand
    arrival_time = timetable_df['Arrival_Time'].copy()
    bins = range(0, 10*60+1, 30)  # create bins of 30 minutes from 0 to 10*60 (total minutes in 10 hours)
    labels = [f'{i/30}' for i in bins[:-1]]  # create labels for the bins in 0,1,2,3... format
    group = pd.DataFrame()
    group['Group'] = pd.cut(arrival_time, bins=bins, labels=labels, include_lowest=True)
    arrival_time = pd.concat([arrival_time, group], axis=1)
    arrival_time['Stop'] = timetable_df['Stop'].copy()
    # find the columns where Stop equal 0 or equal 5
    # only sart stops need to time dependent passenger demand
    start_0 = arrival_time.loc[(arrival_time['Stop'] == 0)].copy() # Altona_A
    start_5 = arrival_time.loc[(arrival_time['Stop'] == 5)].copy() # Ohlsdorf_B
    # Now we know how many start_demand_0/5 for each group of time to create
    # start_demand_0/5 increasing linearly based on Group information in arrival_time
    start_stop_0 = [(passenger_altona_avg + intensity_linear_rate*float(data['Group'])) for index, data in start_0.iterrows()]
    start_stop_5 = [(passenger_ohlsdorf_avg + intensity_linear_rate*float(data['Group'])) for index, data in start_5.iterrows()]

    passenger_demand_0_5 = [np.round(start_stop_0, 0), np.round(start_stop_5, 0)]  # all passenger onboard demand for start stops Altona_A and Ohlsdorf_B

    return passenger_demand_0_5
###########################################################################################################################'#########################################################################################################################

passenger_demand_0_5 = generate_start_passenger_onboard(timetable_df, passenger_altona_avg, passenger_ohlsdorf_avg, intensity_linear_rate, passenger_demand_std)

def passenger_demand_generator(timetable_df, stops, passenger_demand_0_5, changing_rates, load_data):

    load_data.loc[(load_data['Stop'] == 0), 'Passenger_Demand'] = passenger_demand_0_5[0]
    load_data.loc[(load_data['Stop'] == 5), 'Passenger_Demand'] = passenger_demand_0_5[1]

    for train_id in range(max(timetable_df['Train_ID'])+1):
        for stop in stops:
            if stop != 0 and stop != 5:
                if stop == 4 or stop == 9:
                    stop_index = load_data[(load_data['Stop'] == stop) & (load_data['Train_ID'] == train_id)].index
                    load_data.loc[stop_index, ['Passenger_Demand']] = 0
                else:
                    last_index = load_data[(load_data['Stop'] == stop-1) & (load_data['Train_ID'] == train_id)].index
                    prev_demand = (load_data['Passenger_Demand'].loc[last_index].values).astype(float)
                    changing_rate = changing_rates[stop]
                    changing_number = prev_demand * changing_rate
                    passenger_demand = np.maximum(0, np.round(np.random.normal(prev_demand + changing_number, passenger_demand_std), 0))
                    stop_index = load_data[(load_data['Stop'] == stop) & (load_data['Train_ID'] == train_id)].index
                    load_data.loc[stop_index, ['Passenger_Demand']] = passenger_demand
    return init_load_data 

init_load_data = passenger_demand_generator(timetable_df, stops, passenger_demand_0_5, changing_rates, init_load_data) 


# # Debugging and Testing:

# init_load_data.loc[init_load_data['Train_ID'] == 6]
# init_load_data[init_load_data['Stop'] == 1]

# mu_list = []
# for i in range(0, 10):
#     mu = np.mean(init_load_data.loc[init_load_data['Stop'] == i, 'Passenger_Demand'])
#     mu_list.append(mu)


# is_equal = init_load_data.loc[init_load_data['Stop'] == 5, 'Passenger_Demand'].reset_index(drop=True).equals(start_5['Passenger_Demand'].reset_index(drop=True))

