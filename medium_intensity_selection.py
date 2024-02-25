
import pandas as pd
import numpy as np
import logging
import os
############################################################################################
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
# logging.basicConfig(level=logging.CRITICAL + 1)
# DEBUG, INFO, WARNING, ERROR, CRITICAL

############################################################################################

from class_simulator import Transport_Simulator

arrival_intensity_list = Transport_Simulator.test_cargo_time_intensity_set
load_status = {}
STU_status = {}
results = {}
passenger_demand_mode_set = ['constant', 'linear'] #  
for passenger_demand_mode in passenger_demand_mode_set:
    for decision_1 in ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']: 
        for arrival_intensity in arrival_intensity_list:
            load_status[arrival_intensity] = {}
            STU_status[arrival_intensity] = {}
            for seed in range(1925, 1975): # + 50 seeds (1925, 1975)# + 30 seeds (1975, 2005) # 20 seeds (2005, 2025)
            
                test_run = Transport_Simulator(passenger_baseline_intensity_over_time = passenger_demand_mode, 
                                            STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'uniform', 
                                            decision_1_policy = decision_1, decision_2_policy = 'FCFS', 
                                            selection_mode='STU_Time_Intensity_Selection', sensitivity_pattern = None,
                                            set_intensity_medium = arrival_intensity,
                                            operation_time = 180, random_seed = seed) 
                                            # simulation time is for STU arrival, ture simulation time is 75 + 180 = 255

                test_run.run_simulation()
                final_load_status, final_STU_status = test_run.output_delivery_data()
                # load_status[arrival_intensity][seed] = final_load_status
                # STU_status[arrival_intensity][seed] = final_STU_status
            # Seed does not influce the revenue path
            # revenue_result = test_run.get_revenue_data()
            # results[arrival_intensity] = revenue_result




  