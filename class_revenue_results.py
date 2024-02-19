import pandas as pd
import numpy as np
import os
import logging
    
# Only Delay <= 30 still generate revenue, when accepted
# Exceeding simulation time, still waiting status STU, if with any delay, will not generate revenue
# Exceeding simulation time, still waiting status STU, if without delay, handel as rejected, will generate revenue(4.5)
# Rejected requests generate same revenue (4.5) slight lower than the lowest Revenue per STU (5.0) with ÖPNV
# effective revenue of accepted STU, depends on the delay:
# Delay = 0, 100% the revenue of individual STU
# Delay = 0-15, 50% the revenue of individual STU
# Delay = 15-30, 25% the revenue of individual STU
# Delay > 30, 0% the revenue of individual STU

class Revenue_Result:
    revenue_of_rejection_or_uncompletion = 4.5 # slightly lower than lowest STU revenue with ÖPNV
    revenue_rate_of_delay_0_15 = 0.5
    revenue_rate_of_delay_15_30 = 0.25
    revenue_rate_of_delay_gt_30 = 0 
    # end_delay_true_waiting_penalty:   
    # worst case, at the end of simulation time, already delay and still waiting
    delay_0_15_waiting_penalty = 5.0
    delay_15_30_waiting_penalty = 10.0
    delay_gt_30_waiting_penalty = 15.0

    def __init__(self, main_dir, sub_dir, selection_mode: str):
        self.directory = os.path.join(main_dir, sub_dir)
        print(f'join_directory = {self.directory}')
        self.selection_mode = selection_mode

    def calculate_total_revenue_for_instance(self):
        results = []  # List to store all result DataFrames

        if self.selection_mode == 'Policy_Selection':
            # Check if the directory contains any files
            if not any(os.scandir(self.directory)):
                print('No file in this directory')
            else:
                # Iterate over all files in the directory
                for root, dirs, files in os.walk(self.directory):
                    for file in files:
                        # Iterate over all files name start with: 'requests_' in folder
                        if file.startswith('requests_'):  # scanning the file with the correct name
                            # Extract stu intensity, policy, passenger demand mode, stu demand mode, random seed and simulation time from the file name
                            STU_time_intensity = file.split('STU')[1].split('_Policy')[0]
                            policy = file.split('_Policy_')[1].split('_Pa')[0]
                            passenger_demand_mode = file.split('Pa')[1].split('_STU')[0]

                            parts = file.split('STU', 2)  # Split the string into at most 3 parts
                            if len(parts) >= 3:  # If there are at least 3 parts, then there were at least 2 'STU's
                                stu_demand_mode = parts[2].split('_Seed')[0]
                            else:
                                # Handle the case where there were less than 2 'STU's
                                # For example, set stu_demand_mode to None or to an empty string
                                stu_demand_mode = None

                            random_seed = file.split('_Seed')[1].split('_Time')[0]
                            simulation_time = file.split('_Time')[1].split('.csv')[0]
                            file_path = os.path.join(root, file)
                            logging.info(f'Final STU Request File Path:{file_path}')
                            # Calculate total revenue for the file
                            result = self.calculate_total_revenue(file_path, random_seed, simulation_time, STU_time_intensity, policy, passenger_demand_mode, stu_demand_mode)
                            results.append(result)
                        

        elif self.selection_mode == 'STU_Time_Intensity_Selection':
            if not any(os.scandir(self.directory)):
                print('No file in this directory')
            else:
                # Iterate over all files in the directory
                for root, dirs, files in os.walk(self.directory):
                    for file in files:
                        if file.startswith('requests_'):  # scanning the file with the correct name
                            # Extract stu intensity, policy, passenger demand mode, stu demand mode, random seed and simulation time from the file name
                            STU_time_intensity = file.split('STU')[1].split('_Policy')[0]
                            policy = file.split('_Policy_')[1].split('_Pa')[0]
                            passenger_demand_mode = file.split('Pa')[1].split('_STU')[0]

                            parts = file.split('STU', 2)  # Split the string into at most 3 parts
                            if len(parts) >= 3:  # If there are at least 3 parts, then there were at least 2 'STU's
                                stu_demand_mode = parts[2].split('_Seed')[0]
                            else:
                                # Handle the case where there were less than 2 'STU's string
                                # For example, set stu_demand_mode to None or to an empty string
                                stu_demand_mode = None

                            random_seed = file.split('_Seed')[1].split('_Time')[0]
                            simulation_time = file.split('_Time')[1].split('.csv')[0]
                            file_path = os.path.join(root, file)
                            logging.info(f'Final STU Request File Path:{file_path}')
                            # Calculate total revenue for the file
                            result = self.calculate_total_revenue(file_path, random_seed, simulation_time, STU_time_intensity, policy, passenger_demand_mode, stu_demand_mode)
                            results.append(result)

        # Concatenate all result DataFrames together
        final_result = pd.concat(results, ignore_index=True)

        return final_result
    


    def calculate_total_revenue(self, file_path, random_seed, simulation_time, STU_time_intensity, policy, passenger_demand_mode, stu_demand_mode):
        STU_requests_df = pd.read_csv(file_path, index_col = 0)
        
        Passenger_Demand_Mode = passenger_demand_mode
        STU_Demand_Mode = stu_demand_mode
        Policy = policy

        # Define the conditions for each status and delay
        cond_status_0 = STU_requests_df['Status'] == 0   # Rejected
        cond_status_1 = STU_requests_df['Status'] == 1   # Waiting
        cond_status_2 = STU_requests_df['Status'] == 2   # On_Train
        cond_status_3 = STU_requests_df['Status'] == 3   # Delivered
        cond_delay_0 = STU_requests_df['Delay'] == 0     # On time, rejected has delay of np.nan, not counted hier
        cond_delay_nan = STU_requests_df['Delay'].isna()  # Rejected or Arrival to late, excceded simulation time
        cond_delay_0_15 = (STU_requests_df['Delay'] > 0) & (STU_requests_df['Delay'] <= 15)
        cond_delay_15_30 = (STU_requests_df['Delay'] > 15) & (STU_requests_df['Delay'] <= 30)
        cond_delay_gt_30 = STU_requests_df['Delay'] > 30

        # Calculate total revenue and counts for each condition
        delay_0_delivery = ((cond_status_2 | cond_status_3) & cond_delay_0).sum()
        delay_0_15_delivery = ((cond_status_2 | cond_status_3) & cond_delay_0_15).sum()
        delay_15_30_delivery = ((cond_status_2 | cond_status_3) & cond_delay_15_30).sum()
        delay_gt_30_delivery = ((cond_status_2 | cond_status_3) & cond_delay_gt_30).sum()
        delay_0_waiting = (cond_status_1 & cond_delay_0).sum() # waiting without delay
        delay_true_waiting = (cond_status_1 & (cond_delay_0_15 | cond_delay_15_30 | cond_delay_gt_30)).sum() # waiting with delay
        delay_0_15_waiting = (cond_status_1 & cond_delay_0_15).sum() # waiting with delay
        delay_15_30_waiting = (cond_status_1 & cond_delay_15_30).sum() # waiting with delay
        delay_gt_30_waiting = (cond_status_1 & cond_delay_gt_30).sum() # waiting with delay
        delay_nan_waiting = (cond_status_1 & cond_delay_nan).sum() #arrival too late, no delay data
        status_3 = cond_status_3.sum()
        status_2 = cond_status_2.sum()
        status_1 = cond_status_1.sum()
        status_0 = cond_status_0.sum()
        STU_Total = status_0 + status_1 + status_2 + status_3
        STU_Accepted = status_1 + status_2 + status_3
        
        revenue_1 = Revenue_Result.revenue_of_rejection_or_uncompletion * (status_0 + delay_0_waiting + delay_nan_waiting)
        revenue_2 = STU_requests_df.loc[cond_status_2 & cond_delay_0, 'Revenue'].sum()
        revenue_3 = STU_requests_df.loc[cond_status_3 & cond_delay_0, 'Revenue'].sum()
        revenue_4 = Revenue_Result.revenue_rate_of_delay_0_15 * STU_requests_df.loc[(cond_status_2 | cond_status_3) & cond_delay_0_15, 'Revenue'].sum()
        revenue_5 = Revenue_Result.revenue_rate_of_delay_15_30 * STU_requests_df.loc[(cond_status_2 | cond_status_3) & cond_delay_15_30, 'Revenue'].sum()
        penalty_delay_true_waiting = Revenue_Result.delay_0_15_waiting_penalty * delay_0_15_waiting + Revenue_Result.delay_15_30_waiting_penalty * delay_15_30_waiting + Revenue_Result.delay_gt_30_waiting_penalty * delay_gt_30_waiting
        total_revenue = np.round(revenue_1 + revenue_2 + revenue_3 + revenue_4 + revenue_5 - penalty_delay_true_waiting, 6)
        imaginary_revenue = np.round(STU_requests_df['Revenue'].sum(), 6)   # if we get all revenues, without any rejection or delay
        # print(f'revenue_1 = {revenue_1}, revenue_2 = {revenue_2}, revenue_3 = {revenue_3}, revenue_4 = {revenue_4}, revenue_5 = {revenue_5}, total_revenue = {total_revenue}')
        # print(f"status_3 without delay: {STU_requests_df.loc[cond_status_3 & cond_delay_0, 'Revenue'].to_list()} \n R3_total: {sum(STU_requests_df.loc[cond_status_3 & cond_delay_0, 'Revenue'].to_list())}")
        # print('#############################################################################################################')


        result = {
            'Seed_Time_Intensity, Policy': [[random_seed + '_' + simulation_time + '_' + STU_time_intensity, Policy]],
            'Passenger_Demand_Mode, STU_Demand_Mode': [[Passenger_Demand_Mode, STU_Demand_Mode]],
            'STU_Total, Revenue_Total': [[STU_Total, total_revenue]],
            'Imaginary_Revenue, PRT to %': [[imaginary_revenue, np.round(total_revenue / imaginary_revenue, 3)]],
            'Reject_All_Revenue, PRT to %': [[STU_Total * Revenue_Result.revenue_of_rejection_or_uncompletion, 
                                              np.round(total_revenue / (STU_Total * Revenue_Result.revenue_of_rejection_or_uncompletion), 3)]],
            'Delay_0_delivery (% Accepted)': [[delay_0_delivery, np.round(delay_0_delivery / STU_Accepted, 3)]],
            'Delay_0_15_delivery': [[delay_0_15_delivery, np.round(delay_0_15_delivery / STU_Accepted, 3)]],
            'Delay_15_30_delivery': [[delay_15_30_delivery, np.round(delay_15_30_delivery / STU_Accepted, 3)]],
            'Delay_gt_30_delivery': [[delay_gt_30_delivery, np.round(delay_gt_30_delivery / STU_Accepted, 3)]],
            'Delay_0_waiting': [[delay_0_waiting, np.round(delay_0_waiting / STU_Accepted, 3)]],
            'Delay_nan_waiting(late_arrival)': [[delay_nan_waiting, np.round(delay_nan_waiting / STU_Accepted, 3)]],
            'Delay_true_waiting': [[delay_true_waiting, np.round(delay_true_waiting / STU_Accepted, 3)]],
            'Delay_0_15_waiting': [[delay_0_15_waiting, np.round(delay_0_15_waiting / STU_Accepted, 3)]],
            'Delay_15_30_waiting': [[delay_15_30_waiting, np.round(delay_15_30_waiting / STU_Accepted, 3)]],
            'Delay_gt_30_waiting': [[delay_gt_30_waiting, np.round(delay_gt_30_waiting / STU_Accepted, 3)]],
            'Delivered (% Total)': [[status_3, np.round(status_3 / STU_Total, 3)]],
            'On_Train': [[status_2, np.round(status_2 / STU_Total, 3)]],
            'Waiting': [[status_1, np.round(status_1 / STU_Total, 3)]],
            'Rejected': [[status_0, np.round(status_0 / STU_Total, 3)]]
        }
        result = pd.DataFrame(result, index=[0])
        return result
        
## STU_Time_Intensity_Selection:
# intensity = 1.0
# main_dir= r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_constant'
# sub_dir= f'Intensity_{intensity}'
# result = Revenue_Result(main_dir, sub_dir, selection_mode = 'STU_Time_Intensity_Selection')
# final_results = result.calculate_total_revenue_for_instance()
    
# Policy_Selection
# d_1 = 'Available_Train_2'
# d_2 = 'FCFS'
# main_dir = r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_linear'
# sub_dir = f'Policy_{d_1}_{d_2}'
# result = Revenue_Result(main_dir, sub_dir, selection_mode = 'Policy_Selection')
# final_results = result.calculate_total_revenue_for_instance()






