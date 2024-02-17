import pandas as pd
import numpy as np
import logging
from class_simulator import Transport_Simulator
from class_revenue_results import Revenue_Result


class Avg_Train_Load:
    # For Policy:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']
    decision_2_policy_list = ['Random', 'FCFS']
    group = 20 # 20 Seeds
    seed1 = 2005
    passenger_demand_mode_set = ['constant', 'linear']# 

    # For STU_Time_Intensity_Selection:
    arrival_intensity_list = Transport_Simulator.test_cargo_time_intensity_set