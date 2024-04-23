import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
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
excution_time = pd.DataFrame(columns = ['Sensitivity_Pattern', 'Arrival_Over_Station', 'Decision_1', 'Decision_2', 'Seed', 'Execution_Time'])
sensitivity_pattern_set = Transport_Simulator.sensitivity_pattern_set
#['Passenger_Demand_Time_Intensity', 'STU_Demand_Station_Intensity', 'STU_Demand_Time_Intensity'] 
STU_arrival_over_station_set = Transport_Simulator.STU_arrival_over_station_set
# ['uniform', 'hermes_peaks']

for sensitivity in ['STU_Demand_Station_Intensity']:
    for arrival_over_station in ['hermes_peaks']:
        for decision_1 in ['Available_Train_2']: 
            for decision_2 in ['FCFS']:
                for passenger_baseline in ['linear']:
                    for seed in range(1925,2025): 
                        # record current timestamp
                        loop_start = datetime.now()
                    
                        test_run = Transport_Simulator(passenger_baseline_intensity_over_time = passenger_baseline, 
                                                    STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = arrival_over_station, 
                                                    decision_1_policy = decision_1, decision_2_policy = decision_2, 
                                                    selection_mode='Sensitivity_Analysis', sensitivity_pattern = sensitivity, 
                                                    set_intensity_medium = None,
                                                    operation_time = 180, random_seed = seed) 
                                                    # simulation time is for STU arrival, simulation time is 75 + 180 = 255

                        test_run.run_simulation()
                        final_load_status, final_STU_status = test_run.output_delivery_data()
                        # record loop end timestamp
                        loop_end = datetime.now()
                        # find difference loop start and end time and display
                        td = (loop_end - loop_start).total_seconds() * 10**3
                        print(f"The time of execution of loop is : {td:.03f}ms")
                        new_row = pd.DataFrame({'Sensitivity_Pattern': [sensitivity], 'Arrival_Over_Station': [arrival_over_station], 'Passenger_Baseline': [passenger_baseline],
                                                'Decision_1': [decision_1], 'Decision_2': [decision_2], 'Seed': [seed], 'Execution_Time': [td]})
                        excution_time = pd.concat([excution_time, new_row], ignore_index=True)


# # Output excution_time to csv:

dir = rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\{sensitivity_pattern_set[1]}_Sensitivity'
filename = f'execution_time_Station_{STU_arrival_over_station_set[1]}.csv'
full_path = os.path.join(dir, filename)
pd.DataFrame.to_csv(excution_time, full_path, index = False)