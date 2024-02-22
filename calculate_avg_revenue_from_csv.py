import pandas as pd
import numpy as np
import logging
from class_simulator import Transport_Simulator
from class_revenue_results import Revenue_Result


class Avg_Revenue:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']
    decision_2_policy_list = ['Random', 'FCFS']
    passenger_demand_mode_set = ['constant', 'linear']
    arrival_intensity_list = Transport_Simulator.test_cargo_time_intensity_set
    STU_arrival_over_station_set = Transport_Simulator.STU_arrival_over_station_set
    STU_arrival_over_time_set = Transport_Simulator.STU_arrival_over_time_set
    group = 50 # 20 Seeds
    start_seed = 1975 # first seed

    def __init__(self, selection_mode: str, sensitivity_pattern: str):
        self.selection_mode = selection_mode
        self.sensitivity_pattern = sensitivity_pattern

    def output_avg_results(self):
        if self.selection_mode == 'Policy_Selection':
            for passenger_demand_mode in Avg_Revenue.passenger_demand_mode_set:
                main_dir= rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{passenger_demand_mode}'
                for d_1 in Avg_Revenue.decision_1_policy_list:
                    for d_2 in Avg_Revenue.decision_2_policy_list:
                        sub_dir= f'Policy_{d_1}_{d_2}'
                        avg_results = self.process_avg_results(main_dir, sub_dir)

                        avg_results.to_csv(rf'{main_dir}\avg_results_{d_1}_{d_2}.csv', index = False)

        elif self.selection_mode == 'STU_Time_Intensity_Selection':
            for passenger_demand_mode in Avg_Revenue.passenger_demand_mode_set:
                main_dir = rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{passenger_demand_mode}'
                for intensity in Avg_Revenue.arrival_intensity_list:
                    sub_dir= f'Intensity_{intensity}'
                    avg_results = self.process_avg_results(main_dir, sub_dir)

                    avg_results.to_csv(rf'{main_dir}\avg_results_intensity{intensity}.csv', index = False)

        elif self.selection_mode == 'Sensitivity_Analysis':
            main_dir = rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\{self.sensitivity_pattern}_Sensitivity'
            if self.sensitivity_pattern == 'Passenger_Demand_Time_Intensity':
                for passenger_demand_mode in Avg_Revenue.passenger_demand_mode_set:
                    sub_dir= f'Passenger_{passenger_demand_mode}'
                    avg_results = self.process_avg_results(main_dir, sub_dir)
                    print(avg_results)
                    avg_results.to_csv(rf'{main_dir}\avg_results_{sub_dir}.csv', index = False)
            elif self.sensitivity_pattern == 'STU_Demand_Time_Intensity':
                for arrival_over_time in Avg_Revenue.STU_arrival_over_time_set:
                    sub_dir= f'Intensity_{arrival_over_time}'
                    avg_results = self.process_avg_results(main_dir, sub_dir)
                    print(avg_results)
                    avg_results.to_csv(rf'{main_dir}\avg_results_{sub_dir}.csv', index = False)
            elif self.sensitivity_pattern == 'STU_Demand_Station_Intensity':
                for arrival_over_station in Avg_Revenue.STU_arrival_over_station_set:
                    sub_dir= f'Station_{arrival_over_station}'
                    avg_results = self.process_avg_results(main_dir, sub_dir)
                    print(avg_results)
                    avg_results.to_csv(rf'{main_dir}\avg_results_{sub_dir}.csv', index = False)
            else:
                raise ValueError('The sensitivity pattern is not valid')
        else:
            raise ValueError('The selection mode is not valid')


############################################################################################################################################################################ 

    def process_avg_results(self, main_dir, sub_dir):
        result = Revenue_Result(main_dir, sub_dir, selection_mode = self.selection_mode)
        final_results = result.calculate_total_revenue_for_instance()
        # print(final_results)

        results, avg_results = self.initialize_avg_results(final_results)
        avg_results['Seed_Time_Intensity, Policy'] = avg_results['Seed_Time_Intensity, Policy'].apply(self.replace_start_seed)

        avg_results = self.calculate_avg_results(results, avg_results)

        # Now avg_results contains the averaged data
        avg_results.reset_index(drop=True, inplace=True)
        avg_results = avg_results.applymap(self.remove_brackets)

        return avg_results                    

    def initialize_avg_results(self, final_results):
        results = final_results.copy()
        avg_results = pd.DataFrame()

        for col in ['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode']:
            column_data = results[col].tolist()
            chunks = [column_data[i:i + Avg_Revenue.group] for i in range(0, len(column_data), Avg_Revenue.group)]
            avg_column = [chunk[0] for chunk in chunks]
            avg_results[col] = avg_column
        return results, avg_results
    
    def calculate_avg_results(self, results, avg_results):
        # Exclude the string columns 'Seed_Time_Intensity, Policy' and 'Passenger_Demand_Mode, STU_Demand_Mode'
        columns_to_process = [col for col in results.columns if col not in ['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode']]

        for col in columns_to_process:
            # Convert the series to a list of lists
            column_data = results[col].tolist()
            # Group the list into chunks of Avg_Revenue.group
            chunks = [column_data[i:i + Avg_Revenue.group] for i in range(0, len(column_data), Avg_Revenue.group)]
            # Calculate the average for each group
            avg_column = [self.calculate_average(chunk) for chunk in chunks]
            # Add the averaged column to the new dataframe
            avg_results[col] = avg_column
        return avg_results

    def replace_start_seed(self, x):
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], str):
            x[0] = x[0].replace(f'{Avg_Revenue.start_seed}', f'AVG_Of_{Avg_Revenue.group}Seeds')
        else:
            raise ValueError('The input is not a list')
        return x
    def calculate_average(self, chunk):
        avg1 = np.round(sum([x[0] for x in chunk]) / len(chunk), 3)
        avg2 = np.round(sum([x[1] for x in chunk]) / len(chunk), 3)
        return [avg1, avg2]
    def remove_brackets(self, cell):
        if isinstance(cell, list):
            return ', '.join(map(str, cell))
        return cell
    



get_avg_revenue = Avg_Revenue(selection_mode = 'Policy_Selection', sensitivity_pattern = None)
get_avg_revenue.output_avg_results()
    
############################################################################################################################################################################

# # For Policy:
# decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']
# decision_2_policy_list = ['Random', 'FCFS']
# group = 20 # 20 Seeds
# seed1 = 2005
# passenger_demand_mode_set = ['constant', 'linear']#  
# for passenger_demand_mode in passenger_demand_mode_set:
#     main_dir= rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{passenger_demand_mode}'
#     for d_1 in decision_1_policy_list:
#         for d_2 in decision_2_policy_list:
#             sub_dir= f'Policy_{d_1}_{d_2}'
#             result = Revenue_Result(main_dir, sub_dir, selection_mode = 'Policy_Selection')
#             final_results = result.calculate_total_revenue_for_instance()
#             print(final_results)

#             results = final_results.copy()
#             avg_results = pd.DataFrame()

#             for col in ['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode']:
#                 column_data = results[col].tolist()
#                 chunks = [column_data[i:i + group] for i in range(0, len(column_data), group)]
#                 avg_column = [chunk[0] for chunk in chunks]
#                 avg_results[col] = avg_column
#             # for avg_results, col == 'Seed_Time_Intensity, Policy'
#             # replace '2005' with 'AVG_Of_{group}'
#             def replace_seed1(x):
#                 if isinstance(x, list) and len(x) > 0 and isinstance(x[0], str):
#                     x[0] = x[0].replace(f'{seed1}', f'AVG_Of_{group}Seeds')
#                 return x

#             avg_results['Seed_Time_Intensity, Policy'] = avg_results['Seed_Time_Intensity, Policy'].apply(replace_seed1)


#             def calculate_average(group):
#                 avg1 = np.round(sum([x[0] for x in group]) / len(group), 3)
#                 avg2 = np.round(sum([x[1] for x in group]) / len(group), 3)
#                 return [avg1, avg2]

#             # Exclude the columns 'Seed_Time_Intensity, Policy' and 'Passenger_Demand_Mode, STU_Demand_Mode'
#             columns_to_process = [col for col in results.columns if col not in ['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode']]

#             for col in columns_to_process:
#                 # Convert the series to a list of lists
#                 column_data = results[col].tolist()

#                 # Group the list into chunks of 10
#                 chunks = [column_data[i:i + group] for i in range(0, len(column_data), group)]

#                 # Calculate the average for each group
#                 avg_column = [calculate_average(chunk) for chunk in chunks]

#                 # Add the averaged column to the new dataframe
#                 avg_results[col] = avg_column

#             # Now avg_results contains the averaged data
#             avg_results.reset_index(drop=True, inplace=True)

#             def remove_brackets(cell):
#                 if isinstance(cell, list):
#                     return ', '.join(map(str, cell))
#                 return cell

#             avg_results = avg_results.applymap(remove_brackets)

#             avg_results.to_csv(rf'{main_dir}\avg_results_{d_1}_{d_2}.csv', index = False)

# ############################################################################################################

# # For STU_Time_Intensity_Selection:
# arrival_intensity_list = Transport_Simulator.test_cargo_time_intensity_set
# group = 20  # 20 Seeds
# seed1 = 2005
# passenger_demand_mode_set = ['constant', 'linear'] 
# for passenger_demand_mode in passenger_demand_mode_set:
#     main_dir = rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{passenger_demand_mode}'
#     for intensity in arrival_intensity_list:
#         sub_dir= f'Intensity_{intensity}'
#         result = Revenue_Result(main_dir, sub_dir, selection_mode = 'STU_Time_Intensity_Selection')
#         final_results = result.calculate_total_revenue_for_instance()
#         print(final_results)

#         results = final_results.copy()
#         # Create a new dataframe to store the results
#         avg_results = pd.DataFrame()
        
#         # For columns 'Seed_Time_Intensity, Policy' and 'Passenger_Demand_Mode, STU_Demand_Mode'
#         # Group the data into chunks of 10
#         for col in ['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode']:
#             column_data = results[col].tolist()
#             chunks = [column_data[i:i + group] for i in range(0, len(column_data), group)]
#             avg_column = [chunk[0] for chunk in chunks]
#             avg_results[col] = avg_column
#         # for avg_results, col == 'Seed_Time_Intensity, Policy'
#         # replace '2015' with 'AVG_Of_{group}'
#         def replace_seed1(x):
#             if isinstance(x, list) and len(x) > 0 and isinstance(x[0], str):
#                 x[0] = x[0].replace(f'{seed1}', f'AVG_Of_{group}Seeds')
#             return x

#         avg_results['Seed_Time_Intensity, Policy'] = avg_results['Seed_Time_Intensity, Policy'].apply(replace_seed1)


#         def calculate_average(group):
#             avg1 = np.round(sum([x[0] for x in group]) / len(group), 3)
#             avg2 = np.round(sum([x[1] for x in group]) / len(group), 3)
#             return [avg1, avg2]

#         # Exclude the columns 'Seed_Time_Intensity, Policy' and 'Passenger_Demand_Mode, STU_Demand_Mode'
#         columns_to_process = [col for col in results.columns if col not in ['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode']]

#         for col in columns_to_process:
#             # Convert the series to a list of lists
#             column_data = results[col].tolist()

#             # Group the list into chunks of 10
#             chunks = [column_data[i:i + group] for i in range(0, len(column_data), group)]

#             # Calculate the average for each group
#             avg_column = [calculate_average(chunk) for chunk in chunks]

#             # Add the averaged column to the new dataframe
#             avg_results[col] = avg_column

#         # Now avg_results contains the averaged data
#         avg_results.reset_index(drop=True, inplace=True)

#         def remove_brackets(cell):
#             if isinstance(cell, list):
#                 return ', '.join(map(str, cell))
#             return cell

#         avg_results = avg_results.applymap(remove_brackets)

#         avg_results.to_csv(rf'{main_dir}\avg_results_intensity{intensity}.csv', index = False)

