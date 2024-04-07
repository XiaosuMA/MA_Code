import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

import train_timetable_S1 as timetable



# HVV passenger data (2014-2020), dirving from Source File: HVV_Passenger_Data_2014-2020.csv and S1_baseline_passenger_demand.py
passenger_onboard_baseline = 23.025793650793652         # average onboard passenger over stops
passenger_onboard_changing_rate = [-0.08080999569151226, 
                                    -0.040723825937096114, 
                                    0.18614390348987508, 
                                    -0.010219732873761322, 
                                    -1, 
                                    -0.11048685911245149, 
                                    0.07825937096079283, 
                                    0.11358897027143477, 
                                    0.12548039638087038, 
                                    -1]   # mean_passenger_onboard = baseline_onboard * (changing_rate + 1)
                                # changing rate represent onboard passenger intensity at different stops

# Base on both above generate Passenger Onboard Capacity Demand for each stop


class Passenger:
    operation_time = 180
    timetable_df = timetable.timetable_df
    timetable_pivot = timetable.timetable_pivot
    columns_train = ['Train_ID', 'Stop', 'Passenger_Demand', 'Passenger_Extra', 'Passenger_Onboard', 'STU_Onboard', 'Current_Load']
    # baseline_start = passenger_onboard_baseline
    changing_rate = passenger_onboard_changing_rate 
    passenger_stop_std = 1.5
    # passenger_stop_std = 0
    intensity_linear_rate = 2.5
    

    def __init__(self, random_seed: int, passenger_baseline_intensity_over_time: str):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.passenger_baseline_intensity_over_time = passenger_baseline_intensity_over_time
        self.baseline_start = self.initialize_passenger_onboard_baseline()
        self.intensity_linear_rate = self.initialize_intensity_linear_rate()

    def generate_initial_load_data(self):
        baseline_series = self.create_baseline_series() 
        init_load_data = self.initiallize_load_data()
        load_data = self.passenger_demand_generator(baseline_series, init_load_data)
        return load_data
    
    def initialize_passenger_onboard_baseline(self):
        if self.passenger_baseline_intensity_over_time == 'constant':
            baseline_start = passenger_onboard_baseline
        elif self.passenger_baseline_intensity_over_time == 'linear':
            # medium of passenger onboard demand baseline  == original passenger onboard demand baseline
            baseline_start = passenger_onboard_baseline - ((Passenger.operation_time/30)*Passenger.intensity_linear_rate)/2
        else:
            print('Error: passenger_baseline_intensity_over_time should be either constant or linear')
        return baseline_start

    def initialize_intensity_linear_rate(self):
        if self.passenger_baseline_intensity_over_time == 'constant':
            intensity_linear_rate = 0
        elif self.passenger_baseline_intensity_over_time == 'linear':
            intensity_linear_rate = Passenger.intensity_linear_rate   # passenger onboard demand baseline increase 3 every 30 minutes

        else:
            print('Error: passenger_baseline_intensity_over_time should be either constant or linear')
        return intensity_linear_rate
    
    def create_baseline_series(self):
        arrival_time = Passenger.timetable_df['Arrival_Time'].copy()
        bins = range(0, 8*60+1, 30)  # create bins of 30 minutes from 0 to 8*60 (total minutes in 10 hours)
        labels = [f'{i/30}' for i in bins[:-1]]  # create labels for the bins in 0,1,2,3... format
        group = pd.DataFrame()
        group['Group'] = pd.cut(arrival_time, bins=bins, labels=labels, include_lowest=True)
        num_group = group['Group'].nunique()
        # print(f'Number of unique values in group: {num_group}')
        arrival_time = pd.concat([arrival_time, group], axis=1) # every arrival time point got a label of group

        # Now we know the order of baseline_onboard group in terms of time for each row
        # baseline increasing linearly based on Group information in arrival_time
        baseline = []
        first_train_arrival_last_stop_label = 75.0//30
        for index, data in arrival_time.iterrows():
            if float(data['Group']) >= first_train_arrival_last_stop_label:
                row_baseline = self.baseline_start + self.intensity_linear_rate*(float(data['Group']) - first_train_arrival_last_stop_label)
            else:
                row_baseline = self.baseline_start
            baseline.append(row_baseline)

        baseline_series = pd.Series(baseline)
        num_unique_values = baseline_series.nunique()
        # print(f'Baseline series: {baseline_series[0:20]}')
        # print(f'Number of unique values in baseline_series: {num_unique_values}')
        return baseline_series
    
    def initiallize_load_data(self):
        # Initialize DataFrame for train load data
        init_load_data = pd.DataFrame(index = Passenger.timetable_df.index, columns=Passenger.columns_train)
        # Populate DataFrame with initial data
        init_load_data[['Arrival_Time','Train_ID', 'Stop']] = Passenger.timetable_df[['Arrival_Time','Train_ID', 'Stop']]
        init_load_data[['Passenger_Demand', 'Passenger_Extra', 'Passenger_Onboard', 'STU_Onboard', 'Current_Load']] = [np.nan, np.nan, np.nan, np.nan, np.nan]
        init_load_data['Passenger_Onboard'] = np.nan
        init_load_data['STU_Onboard'] = np.nan
        init_load_data.loc[init_load_data['Train_ID'] == 0, 'Passenger_Extra'] = 0
        return init_load_data
    

    def passenger_demand_generator(self, baseline_series, init_load_data):
        # Generate passenger demand for each row in load_data base on baseline passenger onboard demand and stop specific changing rate
        # iterate over init_load_data by index 
        # calculate passenger demand for each row base on same index in baseline_series, 
        # to determing the changing rate for the row, use the init_load_data['Stop'] integer number as stop_index to get the changing rate from changing_rate list
        # all passenger demand in init_load_data at stop are calculated by baseline * (1 + changing_rate[stop_index])
        for index, data in init_load_data.iterrows():
            stop_index = int(data['Stop'])
            if stop_index == 4 or stop_index == 9: # Terminal stops
                init_load_data.loc[index, 'Passenger_Demand'] = 0
            else:
                init_load_data.loc[index, 'Passenger_Demand'] = np.maximum(0, np.round(np.random.normal(baseline_series[index] * (1 + Passenger.changing_rate[stop_index]), Passenger.passenger_stop_std), 0))
        return init_load_data
    

# passenger_instance = Passenger(random_seed=2023, passenger_baseline_intensity_over_time='linear')
# init_load_data = passenger_instance.generate_initial_load_data()

# test_create_baseline_series = passenger_instance.create_baseline_series()
# new_df = pd.concat([Passenger.timetable_df['Arrival_Time'], test_create_baseline_series, init_load_data], axis=1)

# new_df[new_df['Train_ID'] == 0]
# new_df[new_df['Stop'] == 2]
# new_df[new_df['Arrival_Time'] == 75+180]
# new_df[220:240]

# init_load_data[init_load_data['Train_ID'] == 0] 
# init_load_data[init_load_data['Stop'] == 2]

# mu_list = []
# for i in range(0, 10):
#     mu = np.mean(init_load_data.loc[init_load_data['Stop'] == i, 'Passenger_Demand'])
#     mu_list.append(mu)

#####################################################################################################################

# S1_stops = ['Altona A', 'Jungfernstieg A', 'Berliner Tor A', 'Barmbek A', 'Ohlsdorf A', 'Ohlsdorf B', 'Barmbek B', 'Berliner Tor B', 'Jungfernstieg B', 'Altona B']

# means_passenger_onboard = []
# for rate in passenger_onboard_changing_rate:
#     mean_passenger = passenger_onboard_baseline * (1 + rate)
#     means_passenger_onboard.append(mean_passenger)
# # print(means_passenger_onboard)
# bars = plt.bar(S1_stops, means_passenger_onboard, color='gray', alpha=0.7)
# plt.title('Stop specific Means of Onboard Passengers with\nBaseline and Changing Rates', fontsize=8)
# plt.xticks(rotation=30, fontsize=8)  # Rotate x-axis labels for better visibility
# plt.yticks([0, 10, 20, 30], fontsize=8)
# # Add a dashed gray line at passenger_onboard_baseline
# plt.axhline(passenger_onboard_baseline, color='gray', alpha = 0.3, linestyle='--')
# # Annotate the baseline
# centered_text = 'constant\n baseline:\n' + f'{np.round(passenger_onboard_baseline,3)}'
# plt.annotate(centered_text, xy=(9.5, passenger_onboard_baseline), xytext=(-10, -10), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='black', alpha=1)

# # Add passenger_onboard_changing_rate values on top of each bar
# for bar, rate in zip(bars, passenger_onboard_changing_rate):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+0.05, f'{round(rate * 100, 2)}%', ha='center', va='bottom', fontsize=8)
# plt.ylim(bottom = 0, top = 30)
# plt.tight_layout()
# plt.savefig('D:\\Nextcloud\\Data\\MA\\Code\\PyCode_MA\\Outputs\\means_of_onboard_passenger.png', dpi = 300)
# plt.show()