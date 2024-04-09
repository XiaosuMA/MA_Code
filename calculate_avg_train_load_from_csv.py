import pandas as pd
import numpy as np
import logging
from class_simulator import Transport_Simulator
from class_train_load_results import Train_Load_Results


class Avg_Train_Load:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3', 'Available_Train_4', 'Available_Train_5']
    decision_2_policy_list = ['Random', 'FCFS']
    passenger_demand_mode_set = ['constant', 'linear']
    arrival_intensity_list = Transport_Simulator.test_cargo_time_intensity_set
    STU_arrival_over_station_set = Transport_Simulator.STU_arrival_over_station_set
    STU_arrival_over_time_set = Transport_Simulator.STU_arrival_over_time_set
    group = 100 # 50 Seeds
    start_seed = 1925

    def __init__(self, selection_mode: str, sensitivity_pattern: str, mix: bool):
        self.selection_mode = selection_mode
        self.sensitivity_pattern = sensitivity_pattern
        self.mix = mix

    def output_avg_results(self):
        if self.selection_mode == 'Policy_Selection':
            for passenger_demand_mode in Avg_Train_Load.passenger_demand_mode_set:
                main_dir= rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{passenger_demand_mode}'
                for d_1 in Avg_Train_Load.decision_1_policy_list:
                    for d_2 in Avg_Train_Load.decision_2_policy_list:
                        sub_dir= f'Policy_{d_1}_{d_2}'
                        avg_results = self.process_avg_results(main_dir, sub_dir)

                        avg_results.to_csv(rf'{main_dir}\avg_train_load_{d_1}_{d_2}.csv', index = False)

        elif self.selection_mode == 'STU_Time_Intensity_Selection':
            for passenger_demand_mode in Avg_Train_Load.passenger_demand_mode_set:
                main_dir = rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{passenger_demand_mode}'
                for intensity in Avg_Train_Load.arrival_intensity_list:
                    sub_dir= f'Intensity_{intensity}'
                    avg_results = self.process_avg_results(main_dir, sub_dir)

                    avg_results.to_csv(rf'{main_dir}\avg_train_load_intensity{intensity}.csv', index = False)

        elif self.selection_mode == 'Sensitivity_Analysis':
            main_dir = rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\{self.sensitivity_pattern}_Sensitivity'
            if self.sensitivity_pattern == 'Passenger_Demand_Time_Intensity':
                for passenger_demand_mode in Avg_Train_Load.passenger_demand_mode_set:
                    sub_dir= f'Passenger_{passenger_demand_mode}'
                    avg_results = self.process_avg_results(main_dir, sub_dir)
                    print(avg_results)
                    avg_results.to_csv(rf'{main_dir}\avg_train_load_{sub_dir}.csv', index = False)
            elif self.sensitivity_pattern == 'STU_Demand_Time_Intensity':
                for arrival_over_time in Avg_Train_Load.STU_arrival_over_time_set:
                    sub_dir= f'Intensity_{arrival_over_time}'
                    avg_results = self.process_avg_results(main_dir, sub_dir)
                    print(avg_results)
                    avg_results.to_csv(rf'{main_dir}\avg_train_load_{sub_dir}.csv', index = False)
            elif self.sensitivity_pattern == 'STU_Demand_Station_Intensity':
                if self.mix == False:
                    for arrival_over_station in Avg_Train_Load.STU_arrival_over_station_set:
                        sub_dir= f'Station_{arrival_over_station}'
                        avg_results = self.process_avg_results(main_dir, sub_dir)
                        print(avg_results)
                        avg_results.to_csv(rf'{main_dir}\avg_train_load_{sub_dir}.csv', index = False)
                elif self.mix == True:
                        sub_dir= f'Mixed'
                        avg_results = self.process_avg_results(main_dir, sub_dir)
                        print(avg_results)
                        avg_results.to_csv(rf'{main_dir}\avg_train_load_{sub_dir}.csv', index = False)    
            else:
                raise ValueError('The sensitivity pattern is not valid')
        else:
            raise ValueError('The selection mode is not valid')
############################################################################################################################################################################ 

    def process_avg_results(self, main_dir, sub_dir):
        train_load = Train_Load_Results(main_dir, sub_dir, selection_mode = self.selection_mode)
        final_results = train_load.extract_train_load_data()
        # print(final_results)

        results, avg_results = self.initialize_avg_results(final_results)
        avg_results['Seed_Time_Intensity'] = avg_results['Seed_Time_Intensity'].apply(self.replace_start_seed)

        avg_results = self.calculate_avg_results(results, avg_results)

        # Now avg_results contains the averaged data
        avg_results.reset_index(drop=True, inplace=True)

        return avg_results                    

    def initialize_avg_results(self, final_results):
        results = final_results.copy()
        avg_results = pd.DataFrame()

        for col in ['Seed_Time_Intensity', 'Policy', 'Passenger_Demand_Mode', 'STU_Demand_Mode']:
            column_data = results[col].tolist()
            chunks = [column_data[i:i + Avg_Train_Load.group] for i in range(0, len(column_data), Avg_Train_Load.group)]
            avg_column = [chunk[0] for chunk in chunks]
            avg_results[col] = avg_column
        return results, avg_results
    
    def calculate_avg_results(self, results, avg_results):
        # Exclude the string columns 'Seed_Time_Intensity, Policy' and 'Passenger_Demand_Mode, STU_Demand_Mode'
        columns_to_process = [col for col in results.columns if col not in ['Seed_Time_Intensity', 'Policy', 'Passenger_Demand_Mode', 'STU_Demand_Mode']]

        for col in columns_to_process:
            # Convert the series to a list of lists
            column_data = results[col].tolist()
            # Group the list into chunks of Avg_Revenue.group
            chunks = [column_data[i:i + Avg_Train_Load.group] for i in range(0, len(column_data), Avg_Train_Load.group)]
            # Calculate the average for each group
            avg_column = [self.calculate_average(chunk) for chunk in chunks]
            # Add the averaged column to the new dataframe
            avg_results[col] = avg_column
        return avg_results

    def replace_start_seed(self, x):
        if isinstance(x, str):
            x = x.replace(f'{Avg_Train_Load.start_seed}', f'AVG_Of_{Avg_Train_Load.group}Seeds')
        else:
            raise ValueError('The input Seed_Time_Intensity is not a string')
        return x
    def calculate_average(self, chunk):
        avg_value = np.round(sum([x for x in chunk]) / len(chunk), 3)
        
        return avg_value


# instance_avg_train_load = Avg_Train_Load(selection_mode = 'Policy_Selection', sensitivity_pattern = None)
# instance_avg_train_load.output_avg_results()
    
# instance_avg_train_load = Avg_Train_Load(selection_mode = 'STU_Time_Intensity_Selection', sensitivity_pattern = None)
# instance_avg_train_load.output_avg_results()
    
# instance_avg_train_load = Avg_Train_Load(selection_mode = 'Sensitivity_Analysis', sensitivity_pattern = 'Passenger_Demand_Time_Intensity')
# instance_avg_train_load.output_avg_results()

# instance_avg_train_load = Avg_Train_Load(selection_mode = 'Sensitivity_Analysis', sensitivity_pattern = 'STU_Demand_Station_Intensity', mix = False)
# instance_avg_train_load.output_avg_results()

# instance_avg_train_load = Avg_Train_Load(selection_mode = 'Sensitivity_Analysis', sensitivity_pattern = 'STU_Demand_Station_Intensity', mix = True)
# instance_avg_train_load.output_avg_results()