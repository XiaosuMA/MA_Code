# Description: This class is used to extract the train load data from the csv file

import pandas as pd
import numpy as np
import os
import logging
from class_simulator_V0 import Transport_Simulator


class Train_Load_Results:

    def __init__(self, main_dir, sub_dir, selection_mode: str):
        self.directory = os.path.join(main_dir, sub_dir)
        print(f'join_directory = {self.directory}')
        self.selection_mode = selection_mode

    def extract_train_load_data(self):
        results = []  # List to store all result DataFrames

        # Check if the directory contains any files
        if not any(os.scandir(self.directory)):
            print('No file in this directory')
        else:
            # Iterate over all files in the directory
            for root, dirs, files in os.walk(self.directory):
                for file in files:
                    # Iterate over all files name start with: 'load_' in folder
                    if file.startswith('load_'):  # scanning the file with the correct name
                        # Extract stu intensity, policy, passenger demand mode, stu demand mode, random seed and simulation time from the file name
                        STU_time_intensity = file.split('STU')[1].split('_Policy')[0]
                        policy = file.split('_Policy_')[1].split('_Pa')[0]
                        passenger_demand_mode = file.split('Pa')[1].split('_STU')[0]

                        parts = file.split('STU', 2)  # Split the string into at most 3 parts
                        if len(parts) >= 3:  # If there are at least 3 parts, then there were at least 2 'STU's
                            stu_demand_mode = parts[2].split('_Seed')[0]
                        else:
                            # Handle the case where there were less than 2 'STU's
                            # For example, you could set stu_demand_mode to None or to an empty string
                            stu_demand_mode = None

                        random_seed = file.split('_Seed')[1].split('_Time')[0]
                        simulation_time = file.split('_Time')[1].split('.csv')[0]
                        file_path = os.path.join(root, file)
                        logging.info(f'Final STU Request File Path:{file_path}')
                        # Calculate total revenue for the file
                        result = self.process_train_load_data(file_path, random_seed, simulation_time, STU_time_intensity, policy, passenger_demand_mode, stu_demand_mode)
                        results.append(result)

        # Concatenate all result DataFrames together
        final_result = pd.concat(results, ignore_index=True)

        return final_result
    
    def process_train_load_data(self, file_path, random_seed, simulation_time, STU_time_intensity, policy, passenger_demand_mode, stu_demand_mode):
        # Read the file into a DataFrame
        train_load_data = pd.read_csv(file_path, index_col=0)

        train_load_without_terminal_stops = train_load_data[(train_load_data['Stop'] != 4) & (train_load_data['Stop'] != 9) & (train_load_data['Arrival_Time'] > 75)].copy()
        # print(f'{train_load_without_terminal_stops}')
        passenger_extra = train_load_without_terminal_stops['Passenger_Extra'].dropna()
        total_passenger_extra = passenger_extra.sum()
        train_load_percentage = train_load_without_terminal_stops['Current_Load'].dropna()
        average_train_load_percentage = train_load_percentage.mean()
        STU_onboard = train_load_without_terminal_stops['STU_Onboard'].dropna()
        average_STU_onboard = STU_onboard.mean()

        result = {
            'Seed_Time_Intensity, ': [random_seed + '_' + simulation_time + '_' + STU_time_intensity],
            'Policy': [policy],
            'Passenger_Demand_Mode': [passenger_demand_mode],
            'STU_Demand_Mode': [stu_demand_mode],
            'Total_Passenger_Extra': [total_passenger_extra],
            'Average_Train_Load_Percentage': [average_train_load_percentage],
            'Average_STU_Onboard': [average_STU_onboard]
        }
        result = pd.DataFrame(result, index=[0])
        return result
    

# # STU_Time_Intensity_Selection:
# intensity = 1.0
# main_dir= r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant'
# sub_dir= f'Intensity_{intensity}'
# result = Train_Load_Results(main_dir, sub_dir, 'STU_Time_Intensity_Selection')
# final_result = result.extract_train_load_data().head(10)
    
# # Policy_Selection
# d_1 = 'Available_Train_2'
# d_2 = 'FCFS'
# main_dir = r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_linear'
# sub_dir = f'Policy_{d_1}_{d_2}'
# result = Train_Load_Results(main_dir, sub_dir, selection_mode = 'Policy_Selection')
# final_results = result.extract_train_load_data()
    
