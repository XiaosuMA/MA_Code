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
from class_cargo_request import STU_Request
from class_train_functions import Train
from class_policy import Policy
from class_revenue_results import Revenue_Result
from class_simulator import Transport_Simulator


results = {}
passenger_demand_mode_set = ['constant', 'linear'] # 
for passenger_demand_mode in passenger_demand_mode_set:
    for decision_1 in ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']: 
        for decision_2 in ['Random', 'FCFS']:
            for seed in range(1925, 1975): # +50 seeds (1925, 1975) # + 30 seeds (1975, 2005)# 20 seeds, (2005, 2025)
            
                test_run = Transport_Simulator(passenger_baseline_intensity_over_time = passenger_demand_mode, 
                                            STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'uniform', 
                                            decision_1_policy = decision_1, decision_2_policy = decision_2, 
                                            selection_mode='Policy_Selection', sensitivity_pattern = None,
                                            set_intensity_medium = None,
                                            operation_time = 180, random_seed = seed) 
                                            # simulation time is for STU arrival, simulation time is 75 + 180 = 255

                test_run.run_simulation()
                final_load_status, final_STU_status = test_run.output_delivery_data()

    # Seed does not influce the revenue path
    # revenue_result = test_run.get_revenue_data()
    # results[passenger_demand_mode] = revenue_result

                
# from multiprocessing import Pool

# def run_multiple_simulation(args):
#     passenger_demand_mode, decision_1, decision_2, seed = args
#     test_run = Transport_Simulator(passenger_baseline_intensity_over_time = passenger_demand_mode, 
#                                     STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'uniform', 
#                                     decision_1_policy = decision_1, decision_2_policy = decision_2, 
#                                     selection_mode='Policy_Selection', set_intensity_medium = 2.25,
#                                     operation_time = 180, random_seed = seed) 

#     test_run.run_simulation()
#     final_load_status, final_STU_status = test_run.output_delivery_data()

# passenger_demand_mode_set = ['constant', 'linear']
# decision_1_set = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']
# decision_2_set = ['Random', 'FCFS']
# seeds = range(2005, 2025)

# # Create a list of arguments for each simulation
# args = [(passenger_demand_mode, decision_1, decision_2, seed) 
#         for passenger_demand_mode in passenger_demand_mode_set
#         for decision_1 in decision_1_set
#         for decision_2 in decision_2_set
#         for seed in seeds]

# # Create a pool of workers
# with Pool() as p:
#     p.map(run_multiple_simulation, args)

# import multiprocessing

# num_cores = multiprocessing.cpu_count()
# print(f'Number of cores: {num_cores}')