import simpy
import pandas as pd
import numpy as np
import logging


def get_index(df: pd.DataFrame, train_id: int, stop: int):
    """
    This function returns the index of the first row in the dataframe that matches the given train_id and stop.
    If no matching row is found, it returns None.
    """
    index = df[(df['Train_ID'] == train_id) & (df['Stop'] == stop)].index
    if len(index) > 0:
        return int(index[0])
    else:
        return None

#######################################################################################################################

class Train:
    capacity_per_train = 50
    standing_capacity_per_train = 30
    seating_capacity_per_train = capacity_per_train - standing_capacity_per_train

    def __init__(self, env, train_id, stop, 
                 timetable_df, timetable_pivot, 
                 seating_capacity, standing_capacity, 
                 STU_capacity_need, 
                 terminal_stops, start_stops, 
                 load_data, STUs_df):
        self.env = env
        self.train_id = train_id
        self.stop = stop
        self.timetable_df = timetable_df
        self.timetable_pivot = timetable_pivot
        self.seating_capacity = seating_capacity
        self.standing_capacity = standing_capacity
        self.STU_capacity_need = STU_capacity_need
        self.terminal_stops = terminal_stops
        self.start_stops = start_stops
        self.load_data = load_data
        self.STUs_df = STUs_df

#########################################################################################################################   

    def process_train_loading(self):
        logging.info(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info(f'Time: {self.env.now}, Train {self.train_id} arrives Stop {self.stop}.')

        # Unload containers
        self.load_data, self.STUs_df = self.process_STU_unloading()

        # Passenger onboard changing:
        self.load_data = self.process_passenger_boarding()

        # Containers loading:
        self.load_data, self.STUs_df = self.process_STU_loading()

        return self.load_data, self.STUs_df
    
#########################################################################################################################
    
    def process_STU_unloading(self):
        """
        Main function to unload STUs at the current stop and releases the standing capacity.

        Parameters:
        env: The simulation environment.
        train_id (int): The ID of the train.
        stop (int): The current stop.
        standing_capacity: The standing capacity of the train.
        load_data: The load data of the train.
        STUs_df: The dataframe of STUs.

        Returns:
        tuple: The updated load data and STUs dataframe.
        """
        train_stop_index = get_index(self.load_data, self.train_id, self.stop)

        # Select STUs that need to be unloaded at the current stop
        STUs_to_unload = self.STUs_df[
            (self.STUs_df['Stop_d'] == self.stop) &
            (self.STUs_df['Deliver_Train'] == self.train_id) &
            (self.STUs_df['Status'] == 2)
        ].copy()

        # Check if the train has any STUs to unload
        if not STUs_to_unload.empty:
            self.unload_STU(STUs_to_unload)
            self.release_standing_capacity(STUs_to_unload)

            # Update number of STU_onboard
            self.load_data.loc[train_stop_index, 'STU_Onboard'] -= len(STUs_to_unload)
        else:
            logging.debug(f'*Unloading_STU*Time: {self.env.now}, Train {self.train_id} at stop {self.stop} has no STUs to unload.')

        return self.load_data, self.STUs_df


    def unload_STU(self, STUs_to_unload):
        """
        Unloads STUs from the train.

        Parameters:
        env: The simulation environment.
        STUs_df: The dataframe of STUs.
        STUs_to_unload: The STUs to unload.
        train_id (int): The ID of the train.
        stop (int): The current stop.
        """
        for index, stu_data in STUs_to_unload.iterrows():
            # Record the UnloadTime
            self.STUs_df.loc[index, 'Unload_Time'] = self.env.now

            # Record unloaded STU status
            self.STUs_df.loc[index, 'Status'] = 3

            logging.warning(f"*Unloading_STU*Time: {self.env.now}: STU with ID: {stu_data['STU_ID']} is unloaded by Train {self.train_id} at Stop {self.stop}.")


    def release_standing_capacity(self, STUs_to_unload):
        """
        Releases the standing capacity used by the unloaded STUs.

        Parameters:
        env: The simulation environment.
        standing_capacity: The standing capacity of the train.
        train_id (int): The ID of the train.
        STUs_to_unload: The STUs that were unloaded.
        """
        standing_capacity_free = len(STUs_to_unload) * self.STU_capacity_need
        self.standing_capacity[self.train_id].put(standing_capacity_free)

        logging.debug(f'*Unloading_STU*Time: {self.env.now}, After STU unloading: {standing_capacity_free} units of standing capacity are released.')
        logging.debug(f'*Unloading_STU*Time: {self.env.now}, Standing_capacity level: {self.standing_capacity[self.train_id].level}')

########################################################################################################################################################
    
    def process_passenger_boarding(self):   
        """
        Processes passenger boarding for a given train at a given stop.

        Parameters:
        env: The simulation environment.
        train_id (int): The ID of the train.
        stop (int): The current stop.
        seating_capacity: The seating capacity of the train.
        standing_capacity: The standing capacity of the train.
        load_data: The load data of the train.

        Returns:
        DataFrame: The updated load data.
        """

        # Get passenger demand and extra passengers from the load data
        train_stop_index = get_index(self.load_data, self.train_id, self.stop)
        next_train_index = get_index(self.load_data, self.train_id+1, self.stop)
        last_stop_index = get_index(self.load_data, self.train_id, self.stop-1)
        passenger_demand = self.load_data.loc[train_stop_index, 'Passenger_Demand']
        passenger_extra = self.load_data.loc[train_stop_index, 'Passenger_Extra']

        # If passenger demand or extra passengers is not a number, set it to 0
        passenger_demand = 0 if pd.isna(passenger_demand) else passenger_demand
        passenger_extra = 0 if pd.isna(passenger_extra) else passenger_extra

        logging.info(f'Train {self.train_id} at stop {self.stop} has origin passenger demand of {passenger_demand} and passenger extra of {passenger_extra}.')

        # Update real time passenger demand considering extra passengers from last train failed passenger boarding
        total_passenger_demand = passenger_demand + passenger_extra

        # Ensure STUs_standing is 0 if it's NA
        if pd.isna(self.load_data.loc[train_stop_index, 'STU_Onboard']):
            self.load_data.loc[train_stop_index, 'STU_Onboard'] = 0
        STUs_standing = self.load_data.loc[train_stop_index, 'STU_Onboard']

        logging.debug(f'**Double Check(load_data check)**Time: {self.env.now} Train {self.train_id} at Stop: {self.stop} has {STUs_standing} STUs onboard.')

        # If the stop is a terminal stop, only clear all passengers
        if self.stop in self.terminal_stops:
            logging.debug(f'\n ==Eage stop {self.stop}, all out.== \n')
            self.seating_capacity, self.standing_capacity = self.clear_previous_passengers(last_stop_index)
            passenger_onboard = 0
            passenger_failed = 0
            logging.info(f'**Check**: passenger_onboard: {passenger_onboard}, passenger_failed: {passenger_failed}')
            logging.debug(f'\n ==Eage stop {self.stop}, all out== \n')

        # If the stop is a start stop, only load passengers
        elif self.stop in self.start_stops:
            logging.debug(f'*Start_Stop_Before*Train {self.train_id} at stop {self.stop} has: \n free seating capacity of {self.seating_capacity[self.train_id].level} \n free standing capacity of {self.standing_capacity[self.train_id].level}.')
            passenger_onboard, passenger_failed = self.load_passengers(total_passenger_demand)
            logging.info(f'**Check**: passenger_onboard: {passenger_onboard}, passenger_failed: {passenger_failed}')
            if passenger_onboard > 0:
                self.seating_capacity, self.standing_capacity = self.get_capacity(passenger_onboard)

        # If the stop is neither a terminal nor a start stop, clear previous passengers and load new passengers
        else:
            self.seating_capacity, self.standing_capacity = self.clear_previous_passengers(last_stop_index)
            passenger_onboard, passenger_failed = self.load_passengers(total_passenger_demand)
            logging.info(f'**Check**: passenger_onboard: {passenger_onboard}, passenger_failed: {passenger_failed}')
            if passenger_onboard > 0:
                self.seating_capacity, self.standing_capacity = self.get_capacity(passenger_onboard)

        # Update the load data for this train at this stop and the extra passengers for the next train at this stop
        self.load_data.loc[train_stop_index,'Passenger_Onboard'] = passenger_onboard
        self.load_data.loc[next_train_index,'Passenger_Extra'] = passenger_failed

        return self.load_data
   

    def clear_previous_passengers(self, last_stop_index):
        """
        Clears passengers load from the previous stop and puts back the seating and standing capacities to the train.

        Parameters:
        load_data: The load data of the train.
        last_stop_index (int): The index of the last stop.
        seating_capacity: The seating capacity of the train.
        standing_capacity: The standing capacity of the train.
        train_id (int): The ID of the train.
        stop (int): The current stop.

        Returns:
        tuple: The updated seating and standing capacities.
        """
        # Get the number of passengers onboard at the last stop
        previous_passenger_load = self.load_data.loc[last_stop_index, 'Passenger_Onboard']

        # Check if the previous passenger load is NA
        if pd.isna(previous_passenger_load):
            logging.critical(f'Last stop: {self.stop-1}, Train {self.train_id} Passenger_Onboard is nan.')
            raise ValueError('previous_passenger_load is nan')

        # If the previous passenger load exceeds the seating capacity, put back the full seating capacity
        # and the excess as standing capacity puts back
        if previous_passenger_load - Train.seating_capacity_per_train > 0:
            self.seating_capacity[self.train_id].put(Train.seating_capacity_per_train)
            previous_passenger_standing = previous_passenger_load - Train.seating_capacity_per_train
            self.standing_capacity[self.train_id].put(previous_passenger_standing)
            logging.info(f'**Pre_Load**Train {self.train_id}, Prev_Passenger {previous_passenger_load} (Prev_Stop {self.stop-1})')
            logging.info(f'**Clearing_Pa**Stop: {self.stop}, Train {self.train_id} put back {Train.seating_capacity_per_train} seating capacity and {previous_passenger_standing} standing capacity. ')
        # If the previous passenger load does not exceed the seating capacity, puts back the actual load to seating capacity
        else: 
            if previous_passenger_load > 0:
                self.seating_capacity[self.train_id].put(previous_passenger_load)
                logging.info(f'**Pre_Load**Train {self.train_id}, Prev_Passenger {previous_passenger_load} (Prev_Stop {self.stop-1})')
                logging.info(f'**Clearing_Pa**Stop: {self.stop}, Train {self.train_id} put back {previous_passenger_load} seating capacity and 0 standing capacity.')
            else:
                logging.critical(f'Last stop: {self.stop-1}, Train {self.train_id} Passenger_Onboard is 0.')

        return self.seating_capacity, self.standing_capacity

    def load_passengers(self, total_passenger_demand):
        """
        Loads passengers onto the train.

        Parameters:
        env: The simulation environment.
        total_passenger_demand (int): The total number of passengers waiting to board.
        seating_capacity: The seating capacity of the train.
        standing_capacity: The standing capacity of the train.
        train_id (int): The ID of the train.
        stop (int): The current stop.

        Returns:
        tuple: The number of passengers that boarded and the number of passengers that failed to board.
        """
        # Calculate the total available capacity on the train
        total_train_capacity = self.seating_capacity[self.train_id].level + self.standing_capacity[self.train_id].level

        # If the total passenger demand exceeds the total train capacity, some passengers will fail to board
        if total_passenger_demand > total_train_capacity:
            passengers_boarded = total_train_capacity
            passengers_failed_to_board = total_passenger_demand - total_train_capacity
            logging.info(f"**Passenger_Failed**{passengers_failed_to_board} Passengers failed embarking Train {self.train_id} at Stop {self.stop}, must wait for the next train.")
        # If the total passenger demand does not exceed the total train capacity, all passengers will board
        else:
            passengers_boarded = total_passenger_demand
            passengers_failed_to_board = 0

        logging.debug(f'{passengers_boarded} Passengers could embark Train {self.train_id} at Stop {self.stop} (Passenger Demand: {total_passenger_demand}).')

        return passengers_boarded, passengers_failed_to_board

    # Get right amount of Simpy.container capacity:
    def get_capacity(self, passenger_onboard):
        """
        This function adjusts the seating and standing capacities based on the number of onboard passengers.
        """

        if passenger_onboard - self.seating_capacity[self.train_id].level <= 0: # passenger only use seating area
            logging.debug(f'*Passenger_Embarking*Train {self.train_id} at stop {self.stop}: Passenger only use seating area.')
            self.seating_capacity[self.train_id].get(passenger_onboard) # passenger get seating area
            available_capacity = self.seating_capacity[self.train_id].level + self.standing_capacity[self.train_id].level
            logging.debug(f'*Embarked*Train {self.train_id} at stop {self.stop} has: \n free seating capacity of {self.seating_capacity[self.train_id].level} \n free standing capacity of {self.standing_capacity[self.train_id].level}.')
        
        else: # passenger use both seating and standing areas
            logging.debug(f'*Passenger_Embarking*Train {self.train_id} at stop {self.stop}: Passenger use both seating and standing areas.')
            self.seating_capacity[self.train_id].get(Train.seating_capacity_per_train) # passenger get all seating area
            passenger_standing = passenger_onboard - Train.seating_capacity_per_train
            self.standing_capacity[self.train_id].get(passenger_standing) # passenger get some of standing area
            available_capacity = self.standing_capacity[self.train_id].level
            logging.debug(f'*Embarked*Train {self.train_id} at stop {self.stop} has: \n free seating capacity of {self.seating_capacity[self.train_id].level} \n free standing capacity of {self.standing_capacity[self.train_id].level}.')
        
        return self.seating_capacity, self.standing_capacity
    

#########################################################################################################################

    def process_STU_loading(self):

        train_stop_index, TT_next_train_index, next_stop_index = self.get_indices()

        if self.stop == self.terminal_stops[-1]:    # Last terminal stop
            return self.handle_terminal_stop(train_stop_index)

        eligible_STUs = self.select_eligible_STUs()
        # Check if the stop has any STUs waiting to be load:
        if not eligible_STUs.empty:
            # Check if there is any available capacity for STU:
            if self.standing_capacity[self.train_id].level >= self.STU_capacity_need: 
                self.handle_eligible_STUs_with_capacity(eligible_STUs, train_stop_index, TT_next_train_index)
            else: 
                self.handle_no_standing_capacity(eligible_STUs, TT_next_train_index)   # No standing capacity for any STU, re-assing all STUs waiting to next train
        else: 
            logging.debug(f"**No STU waiting**Time: {self.env.now}, train: {self.train_id} at Stop {self.stop} has no STU waiting.")  # No STU is waiting for this train at this stop

        # Initialize the number of STUs onboard with this train_id for arriving next stop
        self.update_for_next_stop(train_stop_index, next_stop_index)
        # Update current load % for leaving the current stop
        self.update_current_load(train_stop_index)

        return self.load_data, self.STUs_df  
    
    def get_indices(self):
        train_stop_index = get_index(self.load_data, self.train_id, self.stop)
        TT_next_train_index = get_index(self.timetable_df, self.train_id+1, self.stop)
        next_stop_index = get_index(self.load_data, self.train_id, self.stop+1) if self.stop != self.terminal_stops[-1] else None
        return train_stop_index, TT_next_train_index, next_stop_index


    def handle_terminal_stop(self, train_stop_index):
        logging.debug(f'Train {self.train_id} finished its journey at stop {self.stop}.')
        self.update_current_load(train_stop_index)
        return self.load_data, self.STUs_df 
    
    
    def select_eligible_STUs(self):
        return self.STUs_df[
            (self.STUs_df['Stop_o'] == self.stop) &
            # (self.STUs_df['PTW_a'] <= self.env.now) &               # sencond decision: assignment, already only assign to trains inside PTW
            # (self.STUs_df['PTW_b'] > self.env.now) &                # reassignment cause only delay, here i use one load queue, so no need to check PTW_b
            (self.STUs_df['Assign_To_Train'] == self.train_id) & (self.STUs_df['Status'] == 1)
        ].copy()


    def handle_eligible_STUs_with_capacity(self, eligible_STUs, train_stop_index, TT_next_train_index):
        eligible_STUs = eligible_STUs.sort_values(by=['Delay', 'Revenue'], ascending=[True, False])  # Load Rules based on how we sort this queue 
                # Sorting STU data with Load Rules, then
                # Load STUs one by one
        for index, STU_data in eligible_STUs.iterrows():
            # Check standing capacity level after each loading
            if self.standing_capacity[self.train_id].level >= self.STU_capacity_need: 
                self.update_STUs_df_for_successful_loading(index)
                self.update_load_data_for_successful_loading(train_stop_index)
                self.update_standing_capacity_for_successful_loading()
                logging.warning(f"**STU Loading**Time: {self.env.now}, train: {self.train_id} load STU with ID: {STU_data['STU_ID']}.")
                logging.debug(f"**After STU loading**Time: {self.env.now}, train: {self.train_id} has \n free standing capacity: {self.standing_capacity[self.train_id].level}.")
            else:
                # If insufficient standing capacity during loading, container re-assign to next coming train, update failure statistics
                self.handle_insufficient_standing_capacity(index, STU_data, TT_next_train_index)
                logging.warning(f"**STU Re-assigned**STU ID: {self.STUs_df.loc[index, 'STU_ID']} is Re-assigned to Next Train: {self.STUs_df.loc[index, 'Assign_To_Train']}, Processing Delay {self.STUs_df.loc[index, 'Delay']}.")
                

    def update_STUs_df_for_successful_loading(self, index):
        self.STUs_df.loc[index, 'Load_Time'] = self.env.now
        self.STUs_df.loc[index, 'Delay'] = 0 if pd.isna(self.STUs_df.loc[index, 'Delay']) else self.STUs_df.loc[index, 'Delay']
        self.STUs_df.loc[index, 'Deliver_Train'] = self.train_id
        self.STUs_df.loc[index, 'Failed_Loading'] = 0 if pd.isna(self.STUs_df.loc[index, 'Failed_Loading']) else self.STUs_df.loc[index, 'Failed_Loading']
        self.STUs_df.loc[index, 'Status'] = 2


    def update_load_data_for_successful_loading(self, train_stop_index):
        self.load_data.loc[train_stop_index, 'STU_Onboard'] += 1


    def update_standing_capacity_for_successful_loading(self):
        self.standing_capacity[self.train_id].get(self.STU_capacity_need)


    def handle_insufficient_standing_capacity(self, index, STU_data, TT_next_train_index):
        logging.debug(f"**STU Loading**Time: {self.env.now}, train: {self.train_id} at Stop {self.stop} can not load STU with ID: {STU_data['STU_ID']}.")
        self.STUs_df.loc[index, 'Assign_To_Train'] += 1
        self.STUs_df.loc[index, 'Failed_Loading'] = 0 if pd.isna(self.STUs_df.loc[index, 'Failed_Loading']) else self.STUs_df.loc[index, 'Failed_Loading']
        self.STUs_df.loc[index, 'Failed_Loading'] += 1
        logging.debug(f'**Failed Loading**STUs reassigned \n  {self.STUs_df.loc[index]}')
        next_arrival = self.timetable_df.loc[TT_next_train_index, 'Arrival_Time']
        PTW_b = STU_data['PTW_b']
        self.STUs_df.loc[index, 'Delay'] = next_arrival - PTW_b if next_arrival - PTW_b > 0 else 0


    def handle_no_standing_capacity(self, eligible_STUs, TT_next_train_index):
        logging.warning(f"**No Standing Capacity**Time: {self.env.now}, train: {self.train_id} at Stop {self.stop} has NO standing capacity for ANY STU loading.")
        self.STUs_df.loc[eligible_STUs.index, 'Assign_To_Train'] += 1
        # self.STUs_df.loc[eligible_STUs.index, 'Failed_Loading'] = 0 if pd.isna(self.STUs_df.loc[eligible_STUs.index, 'Failed_Loading']) else self.STUs_df.loc[eligible_STUs.index, 'Failed_Loading']
        self.STUs_df.loc[eligible_STUs.index, 'Failed_Loading'].fillna(0, inplace=True)
        self.STUs_df.loc[eligible_STUs.index, 'Failed_Loading'] += 1
        next_arrival = self.timetable_df.loc[TT_next_train_index, 'Arrival_Time']
        delays = next_arrival - eligible_STUs['PTW_b']   # Calculate the delay for STUs waiting
        delays[delays < 0] = 0 # handeling negative delays == no delay
        self.STUs_df.loc[eligible_STUs.index, 'Delay'] = delays  # Assign the calculated delays to the 'Delay' column of the STUs_df DataFrame
        # Select 'STU_ID' and 'Delay' and 'Failed_Loading' columns for all reassigned STUs
        reassigned_STUs = self.STUs_df.loc[eligible_STUs.index, ['STU_ID', 'Delay', 'Failed_Loading']]
        logging.warning(f'**STU Loading**All reassigned_STUs: \n {reassigned_STUs.to_string(index=False)}')


    def update_for_next_stop(self, train_stop_index, next_stop_index):
        self.load_data.loc[next_stop_index, 'STU_Onboard'] = self.load_data.loc[train_stop_index, 'STU_Onboard'].copy()
        STUs_standing = self.load_data.loc[train_stop_index, 'STU_Onboard']
        logging.debug(f'**Double Check(load_data check)**Time: {self.env.now} Train {self.train_id} at Stop: {self.stop} has {STUs_standing} STUs onboard.')
        logging.debug(f"**Double Check(load_data check)**Next Stop: {self.stop+1} has initial: {self.load_data.loc[next_stop_index, 'STU_Onboard']} STUs onboard.")

    def update_current_load(self, train_stop_index):
        current_load = np.round(1-(self.seating_capacity[self.train_id].level + self.standing_capacity[self.train_id].level)/Train.capacity_per_train, 3)
        self.load_data.loc[train_stop_index, 'Current_Load'] = current_load

########################################################################################################################################################