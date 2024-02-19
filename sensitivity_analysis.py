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
sensitivity_pattern_set = Transport_Simulator.sensitivity_pattern_set
#['Passenger_Demand_Time_Intensity', 'STU_Demand_Time_Intensity', 'STU_Demand_Station_Intensity'] 
STU_arrival_over_station_set = Transport_Simulator.STU_arrival_over_station_set
# ['uniform', 'hermes_peaks']

for sensitivity in [sensitivity_pattern_set[2]]:
    for arrival_over_station in [STU_arrival_over_station_set[1]]:
        for decision_1 in ['Available_Train_2']: 
            for decision_2 in ['FCFS']:
                for seed in range(2005, 2025): # 20 seeds
                
                    test_run = Transport_Simulator(passenger_baseline_intensity_over_time = 'constant', 
                                                STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = arrival_over_station, 
                                                decision_1_policy = decision_1, decision_2_policy = decision_2, 
                                                selection_mode='Sensitivity_Analysis', sensitivity_pattern = sensitivity, 
                                                set_intensity_medium = None,
                                                operation_time = 180, random_seed = seed) 
                                                # simulation time is for STU arrival, simulation time is 75 + 180 = 255

                    test_run.run_simulation()
                    final_load_status, final_STU_status = test_run.output_delivery_data()