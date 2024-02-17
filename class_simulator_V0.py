import simpy
import pandas as pd
import numpy as np
import logging
import os
############################################################################################
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
# logging.basicConfig(level=logging.CRITICAL + 1)
# DEBUG, INFO, WARNING, ERROR, CRITICAL

# Set up logging
# level = logging.DEBUG
# level = logging.INFO
# level = logging.WARNING
# level = logging.ERROR
level = logging.CRITICAL
logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(level)

class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == level

logger.addFilter(DebugFilter())
############################################################################################
import train_timetable_S1 as Timetable
from class_passenger_init_load_data import Passenger
from class_cargo_request_V1 import STU_Request
from class_train_functions_V0 import Train
from class_policy_V0 import Policy
from class_revenue_results_V1 import Revenue_Result

############################################################################################


class Transport_Simulator:
    # constants
    STU_capacity_need = 4
    terminal_stops = [4, 9]
    start_stops = [0, 5]
    timetable = Timetable.timetable_df.copy()
    timetable_pivot = Timetable.timetable_pivot.copy()
    train_0_arrival_last_stop_time = 75.0
    test_cargo_time_intensity_set  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    cargo_time_intensity_set = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

    def __init__(self, passenger_baseline_intensity_over_time: str, 
                 STU_arrival_over_time: str, STU_arrival_over_station: str, 
                 decision_1_policy : str, decision_2_policy : str, 
                 selection_mode: str, set_intensity_medium: float,
                 random_seed: int,
                 operation_time: int):
        self.env = simpy.Environment()
        self.df_store = simpy.Store(self.env)

        self.decision_1_policy = decision_1_policy
        self.decision_2_policy = decision_2_policy
        self.selection_mode = selection_mode
        self.set_intensity_medium = set_intensity_medium
        self.random_seed = random_seed 
        # Set a random seed for reproducibility
        np.random.seed(self.random_seed)
        self.simulation_time = int(operation_time + Transport_Simulator.train_0_arrival_last_stop_time)
        self.num_trains = len(Transport_Simulator.timetable_pivot)
        
        # demand mode parameters
        self.passenger_baseline_intensity_over_time = passenger_baseline_intensity_over_time
        self.STU_arrival_over_time = STU_arrival_over_time
        self.STU_arrival_over_station = STU_arrival_over_station
        self.avg_cargo_intensity = np.mean(Transport_Simulator.cargo_time_intensity_set)

        self.seating_capacity, self.standing_capacity = self.initialize_capacities() # initializing seating and standing capacities

        self.passenger_instance, self.stu_request_instance = self.instanciate_instance_of_data() # instanciate passenger and STU request instances
        self.policy = self.instanciate_policies() # instanciate decision policies

        init_load_data, init_STU_requests_df = self.initialize_data() # initializing dataset for simulation
        
        self.df_store.put((init_load_data, init_STU_requests_df)) # put the initial data into the store
        
        
    def instanciate_instance_of_data(self):
        # instanciate passenger and STU request instances for simulation
        passenger_instance = Passenger(self.random_seed, self.passenger_baseline_intensity_over_time)

        if self.selection_mode == 'STU_Time_Intensity_Selection':
            stu_request_instance = STU_Request(STU_arrival_over_time = self.STU_arrival_over_time, 
                                               STU_arrival_over_station = self.STU_arrival_over_station, 
                                               random_seed = self.random_seed, simulation_time = self.simulation_time, 
                                               intensity_medium = self.set_intensity_medium)
        elif self.selection_mode == 'Policy_Selection':
            stu_request_instance = STU_Request(STU_arrival_over_time = self.STU_arrival_over_time, 
                                               STU_arrival_over_station = self.STU_arrival_over_station, 
                                               random_seed = self.random_seed, simulation_time = self.simulation_time, 
                                               intensity_medium = self.avg_cargo_intensity)
        else:
            raise ValueError("Invalid selection_mode: Not defined selection_mode.")

        return passenger_instance, stu_request_instance

    def initialize_capacities(self): # initializing seating and standing capacities for simulation
        seating_capacities = {train_id: simpy.Container(self.env, init=Train.seating_capacity_per_train) for train_id in range(self.num_trains)}
        standing_capacities = {train_id: simpy.Container(self.env, init=Train.standing_capacity_per_train) for train_id in range(self.num_trains)}
        return seating_capacities, standing_capacities

    def instanciate_policies(self):
        # instanciate decision policies for simulation
        policy = Policy(self.decision_1_policy, self.decision_2_policy, self.random_seed)
        policy.log_policy()
        return policy
    

    def initialize_data(self):
        # initializing dataset for simulation
        init_load_data = self.passenger_instance.generate_initial_load_data()
        init_STU_requests_df = self.stu_request_instance.generate_STU_requests_df()

        return init_load_data, init_STU_requests_df


    def check_STU_release_and_train_arrival(self):
        load_data, STUs_df = yield self.df_store.get()
        logging.debug(f'############################################################################################')
        logging.debug(f'Time: {self.env.now}, Start check_STU_release_and_train_arrival.')
        STU_release = STUs_df[STUs_df['Arrival_Time'] == self.env.now].copy()
        train_arrives = Transport_Simulator.timetable[Transport_Simulator.timetable['Arrival_Time'] == self.env.now].copy()
        
        if STU_release.empty and train_arrives.empty:
            yield self.df_store.put((load_data, STUs_df))
            logging.debug(f'Time: {self.env.now}, No STU release and No train arrival.')
        
        else:
            yield self.df_store.put((load_data, STUs_df))
            if not STU_release.empty:
                yield self.env.process(self.process_STU_releasing(STU_release))
            else:
                logging.debug(f'Time: {self.env.now}, No STU release.')
            if not train_arrives.empty:    
                yield self.env.process(self.process_train_arrival())
            else:
                logging.debug(f'Time: {self.env.now}, No train arrival.')

    def process_STU_releasing(self, STU_release):
        # Get the initial DataFrames from the store
        init_load_data, STU_requests_df = yield self.df_store.get()
        
        logging.debug(f'Time: {self.env.now}, STU_releasing function called.')
        load_data = init_load_data.copy()
        STUs_df = STU_requests_df.copy()
        # STU_release = STUs_df[STUs_df['Arrival_Time'] == self.env.now].copy()
        
        if not STU_release.empty:
            for index, STU_data in STU_release.iterrows():
                logging.debug(f"Time: {self.env.now}, STU with ID: {STU_data['STU_ID']} is released.")
                id = STU_data['STU_ID']
                PTW_a = STU_data['PTW_a']
                PTW_b = STU_data['PTW_b']
                stop_o = STU_data['Stop_o']
                revenue = STU_data['Revenue']
                # possible trains for STU request
                get_trains = Transport_Simulator.timetable_pivot.loc[(Transport_Simulator.timetable_pivot[stop_o]>= max(PTW_a, self.env.now))&
                                                                     (Transport_Simulator.timetable_pivot[stop_o]<= PTW_b), stop_o]
                get_trains = get_trains.index.tolist()
                logging.warning(f"STU with ID: {STU_data['STU_ID']} can be originally assigned to Trains {get_trains}, Revenue {revenue}.")

                # 1.Decision: Accept or reject STU request
                decision_1 = self.policy.make_decision_1(get_trains, revenue) # 1. Decision
                STUs_df.loc[index, 'Status'] = decision_1
                
                if decision_1 != 0: # 2. Decision: Assign a train for STU request, sequential decision after accpetance
                    logging.warning(f"Time: {self.env.now}, STU with ID: {STU_data['STU_ID']} is accepted.")

                    if len(get_trains) == 0:
                        get_trains = Transport_Simulator.timetable_pivot.loc[(Transport_Simulator.timetable_pivot[stop_o]>= max(PTW_a, self.env.now))&
                                                                             (Transport_Simulator.timetable_pivot[stop_o]<= PTW_b + 30), stop_o]
                        get_trains = get_trains.index.tolist()
                        
                    decision_2 = self.policy.make_decision_2(get_trains) # 2. Decision
                    STUs_df.loc[index, 'Assign_To_Train'] = decision_2
                    logging.warning(f"Time: {self.env.now}, STU with ID: {STU_data['STU_ID']} is assigned to Train {decision_2}.")
                else:
                    if decision_1 == 0:
                        logging.warning(f"Time: {self.env.now}, STU with ID: {STU_data['STU_ID']} is rejected.")
                    else:
                        raise ValueError("Invalid decision_1")
        
            # Put the updated DataFrames back in the store
            yield self.df_store.put((load_data, STUs_df))
        else:
            yield self.df_store.put((load_data, STUs_df))
            yield self.env.timeout(0)


    def process_train_arrival(self):
    
        logging.debug(f'Time: {self.env.now}, Train_arrival function called.')
        train_arrives = Transport_Simulator.timetable[Transport_Simulator.timetable['Arrival_Time'] == self.env.now].copy()
        
        logging.debug(f'Time: {self.env.now}, train_arrives: \n{train_arrives}')
        for index, train_data in train_arrives.iterrows():
            # Get the latest data from the store
            load_data, STUs_df = yield self.df_store.get()

            train_id = train_data['Train_ID']
            stop = train_data['Stop']
            one_train = Train(self.env, train_id, stop, Transport_Simulator.timetable, Transport_Simulator.timetable_pivot, self.seating_capacity, self.standing_capacity, Transport_Simulator.STU_capacity_need, Transport_Simulator.terminal_stops, Transport_Simulator.start_stops, load_data, STUs_df)
            load_data, STUs_df = one_train.process_train_loading()

            # Put the updated data back into the store
            yield self.df_store.put((load_data, STUs_df))
        yield self.env.timeout(0)
        

    def process_task(self):
        for run in range(self.simulation_time):
            yield self.env.process(self.check_STU_release_and_train_arrival())
            yield self.env.timeout(1)  # Wait for 1 time unit before starting the next process

    def run_simulation(self):

        self.env.process(self.process_task())
        self.env.run(until=self.simulation_time)
        logging.critical(f'**************************************************************************************')
        logging.critical(f'Time: {self.env.now}, Simulation with Seed {self.random_seed} and STU intensity {np.round(1/self.stu_request_instance.initialize_STU_arrival_interval(), 3)} finished.')
        logging.critical(f'**************************************************************************************')

############################################################################################################################################################################

    def directory_of_selection_mode(self):
        if self.selection_mode == 'Policy_Selection':
            if self.passenger_baseline_intensity_over_time == 'constant':
                main_dir = r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_constant'
                sub_dir = f'Policy_{self.decision_1_policy}_{self.decision_2_policy}'
            elif self.passenger_baseline_intensity_over_time == 'linear':
                main_dir = r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_linear'
                sub_dir = f'Policy_{self.decision_1_policy}_{self.decision_2_policy}'
            else:
                raise ValueError("Invalid passenger_baseline_intensity_over_time") 
            
        elif self.selection_mode == 'STU_Time_Intensity_Selection':     
            STU_time_intensity = np.round(1/self.stu_request_instance.initialize_STU_arrival_interval(), 3) # Get value of STU_time_intensity
            if STU_time_intensity in Transport_Simulator.test_cargo_time_intensity_set:
                if self.passenger_baseline_intensity_over_time == 'constant':
                    main_dir = r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant'
                    sub_dir = f'Intensity_{STU_time_intensity}'
                elif self.passenger_baseline_intensity_over_time == 'linear':
                    main_dir = r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear'
                    sub_dir = f'Intensity_{STU_time_intensity}'
                else:
                    raise ValueError("Invalid passenger_baseline_intensity_over_time")
            else:
                raise ValueError("Invalid STU_time_intensity")
            
        elif self.selection_mode == 'Sensitivity_Analysis':
            pass
        else:
            raise ValueError("Invalid selection_mode")
        return main_dir, sub_dir


    def output_delivery_data(self):

        main_dir, sub_dir = self.directory_of_selection_mode()

        # Get the final version of the DataFrames
        final_load_data, final_STU_requests_df = self.env.run(self.df_store.get())
        # Get Parameters for file name   
        STU_time_intensity = np.round(1/self.stu_request_instance.initialize_STU_arrival_interval(), 3)
        
        policy = f'{self.decision_1_policy}_{self.decision_2_policy}'
        passenger_demand_mode = self.passenger_baseline_intensity_over_time
        STU_demand_mode = self.STU_arrival_over_time+'_'+self.STU_arrival_over_station
        random_seed = self.random_seed
        simulation_time = self.simulation_time

        # Combine the main directory and subdirectory
        full_dir = os.path.join(main_dir, sub_dir)

        # Create the directory if it doesn't exist
        # The exist_ok=True parameter means the function will not raise an error if the directory already exists
        os.makedirs(full_dir, exist_ok=True)

        if self.selection_mode == 'Policy_Selection':
            file_name = f'STU{STU_time_intensity}_Policy_{policy}_Pa{passenger_demand_mode}_STU{STU_demand_mode}_Seed{random_seed}_Time{simulation_time}'
        elif self.selection_mode == 'STU_Time_Intensity_Selection':
            file_name = f'STU{STU_time_intensity}_Policy_{policy}_Pa{passenger_demand_mode}_STU{STU_demand_mode}_Seed{random_seed}_Time{simulation_time}'
        elif self.selection_mode == 'Sensitivity_Analysis':
            pass
        else:
            raise ValueError("Invalid selection_mode")
        
        # Define the paths for the CSV files
        load_data_path = os.path.join(full_dir, f'load_data_{file_name}.csv')
        STU_requests_path = os.path.join(full_dir, f'requests_{file_name}.csv')
        logging.warning(f'File Full Name: requests_{file_name}.csv')

        # Save the DataFrames as CSV files
        final_load_data.to_csv(load_data_path)
        final_STU_requests_df.to_csv(STU_requests_path)

        return final_load_data, final_STU_requests_df
    
    def get_revenue_data(self):
        main_dir, sub_dir = self.directory_of_selection_mode()

        revenue_instance = Revenue_Result(main_dir, sub_dir, self.selection_mode)
        revenue_result = revenue_instance.calculate_total_revenue_for_instance()
        return revenue_result
    

# test_run = Transport_Simulator(passenger_baseline_intensity_over_time = 'constant', 
#                                 STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'uniform', 
#                                decision_1_policy = 'Available_Train_1', decision_2_policy = 'FCFS', 
#                                selection_mode='STU_Time_Intensity_Selection', set_intensity_medium = 2.0,
#                                operation_time = 180, random_seed = 2023)


# test_run = Transport_Simulator(passenger_baseline_intensity_over_time = 'linear', 
#                                 STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'uniform', 
#                                decision_1_policy = 'Available_Train_2', decision_2_policy = 'FCFS', 
#                                selection_mode='Policy_Selection', set_intensity_medium = 2.25,
#                                operation_time = 180, random_seed = 2023)

# test_run = Transport_Simulator(passenger_baseline_intensity_over_time = 'constant', 
#                                 STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'uniform', 
#                                decision_1_policy = 'Available_Train_2', decision_2_policy = 'Random', 
#                                selection_mode='Policy_Selection', set_intensity_medium = 2.25,
#                                operation_time = 180, random_seed = 2020)

# test_run.run_simulation()
# final_load_status, final_STU_status = test_run.output_delivery_data()
# revenue_result = test_run.get_revenue_data()


# print(f'Train_0 Load_data:')
# print(final_load_status[final_load_status['Train_ID'] == 0])
# print(f'Train_1 Load_data:')
# print(final_load_status[final_load_status['Train_ID'] == 1])
# print(f'Train_2 Load_data:')
# print(final_load_status[final_load_status['Train_ID'] == 2])
# print(f'Train_3 Load_data:')
# print(final_load_status[final_load_status['Train_ID'] == 3])
# print(f'Train_4 Load_data:')
# print(final_load_status[final_load_status['Train_ID'] == 4])
# print(f'Train_5 Load_data:')
# print(final_load_status[final_load_status['Train_ID'] == 5])
# print(f'STU Status for delay:')
# print(final_STU_status[final_STU_status['Delay'] > 0])
# print(f'STU Status for rejected:')
# print(final_STU_status[final_STU_status['Status'] == 0])
# print(f'STU Status for waiting:')
# print(final_STU_status[final_STU_status['Status'] == 1])
# print(f'STU Status for on train:')
# print(final_STU_status[final_STU_status['Status'] == 2])
# print(f'STU Status for completed:')
# print(final_STU_status[final_STU_status['Status'] == 3])
