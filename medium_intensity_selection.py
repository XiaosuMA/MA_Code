
import pandas as pd
import numpy as np
import logging
import os
############################################################################################
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
# logging.basicConfig(level=logging.CRITICAL + 1)
# DEBUG, INFO, WARNING, ERROR, CRITICAL

############################################################################################
from class_passenger_init_load_data import Passenger
from class_cargo_request_V1 import STU_Request
from class_train_functions_V0 import Train
from class_policy_V0 import Policy
from class_revenue_results_V1 import Result
from class_simulator_V0 import Transport_Simulator

arrival_intensity_list = Transport_Simulator.test_cargo_time_intensity_set
load_status = {}
STU_status = {}
results = {}
passenger_demand_mode_set = ['constant', 'linear']
for passenger_demand_mode in passenger_demand_mode_set:
    for decision_1 in ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']: 
        for arrival_intensity in arrival_intensity_list:
            load_status[arrival_intensity] = {}
            STU_status[arrival_intensity] = {}
            for seed in range(2005, 2015):
            
                test_run = Transport_Simulator(passenger_baseline_intensity_over_time = passenger_demand_mode, 
                                            STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'uniform', 
                                            decision_1_policy = decision_1, decision_2_policy = 'FCFS', 
                                            selection_mode='STU_Time_Intensity_Selection', set_intensity_medium = arrival_intensity,
                                            operation_time = 180, random_seed = seed) 
                                            # simulation time is for STU arrival, ture simulation time is 75 + 180 = 255

                test_run.run_simulation()
                final_load_status, final_STU_status = test_run.output_delivery_data()
                # load_status[arrival_intensity][seed] = final_load_status
                # STU_status[arrival_intensity][seed] = final_STU_status
            # Seed does not influce the revenue path
            # revenue_result = test_run.get_revenue_data()
            # results[arrival_intensity] = revenue_result

# print(results[0.5])
# print(results[1])
# print(results[1.5])
# print(results[2])
# print(results[2.5])
# print(results[3])

# results_p5 = pd.DataFrame(results[0.5])
# results_p5.to_csv('D:/Nextcloud/Data/MA/Code/PyCode_MA/Outputs/STU_Time_Intensity_Selection_Output/revenue_results_0.5_medium_intensity_selection.csv')
# results_1 = pd.DataFrame(results[1])
# results_1.to_csv('D:/Nextcloud/Data/MA/Code/PyCode_MA/Outputs/STU_Time_Intensity_Selection_Output/revenue_results_1.0_medium_intensity_selection.csv')
# results_1p5 = pd.DataFrame(results[1.5])
# results_1p5.to_csv('D:/Nextcloud/Data/MA/Code/PyCode_MA/Outputs/STU_Time_Intensity_Selection_Output/revenue_results_1.5_medium_intensity_selection.csv')
# results_2 = pd.DataFrame(results[2])
# results_2.to_csv('D:/Nextcloud/Data/MA/Code/PyCode_MA/Outputs/STU_Time_Intensity_Selection_Output/revenue_results_2.0_medium_intensity_selection.csv')
# results_2p5 = pd.DataFrame(results[2.5])
# results_2p5.to_csv('D:/Nextcloud/Data/MA/Code/PyCode_MA/Outputs/STU_Time_Intensity_Selection_Output/revenue_results_2.5_medium_intensity_selection.csv')
# results_3 = pd.DataFrame(results[3])
# results_3.to_csv('D:/Nextcloud/Data/MA/Code/PyCode_MA/Outputs/STU_Time_Intensity_Selection_Output/revenue_results_3.0_medium_intensity_selection.csv')



  