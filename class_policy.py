import simpy
import pandas as pd
import numpy as np
import logging       


class Policy:
    rejection_revenue = 4.5 

    def __init__(self, decision_1_policy: str, decision_2_policy: str, random_seed: int):
        self.decision_1_policy = decision_1_policy
        self.decision_2_policy = decision_2_policy
        self.random_seed = random_seed
        np.random.seed(self.random_seed)


    def log_policy(self):
        logging.critical(f'*******************************************************************************')
        if self.decision_1_policy == 'Available_Train_1':
            logging.critical(f'Policy 1: Accept, if there exist at least one {self.decision_1_policy} for on time delivery.')
        elif self.decision_1_policy == 'Available_Train_2':
            logging.critical(f'Policy 1: Accept, if there exist at least two {self.decision_1_policy} for on time delivery.')

        elif self.decision_1_policy == 'Available_Train_2_Or_Revenue':
            logging.critical(f'Policy 1: Accept, if there exist at least two {self.decision_1_policy} for on time delivery Or revenue greater equal two times revenue of rejection.')
        elif self.decision_1_policy == 'Available_Train_3':
            logging.critical(f'Policy 1: Accept, if there exist at least three {self.decision_1_policy} for on time delivery.')
        elif self.decision_1_policy == 'Accept_All':
            logging.critical(f'Policy 1: Accept all STU requests.')
        elif self.decision_1_policy == 'Available_Train_4':
            logging.critical(f'Policy 1: Accept, if there exist at least four {self.decision_1_policy} for on time delivery.')
        elif self.decision_1_policy == 'Available_Train_5':
            logging.critical(f'Policy 1: Accept, if there exist at least five {self.decision_1_policy} for on time delivery.')
        else:
            print(f'Invalid decision_1_policy')
        
        if self.decision_2_policy == 'FCFS':
            logging.critical(f'Policy 2: Assign based on Train First Come Assignment.')
        elif self.decision_2_policy == 'Random':
            logging.critical(f'Policy 2: Assign based on Train Random Assignment.')
        else:
            print(f'Invalid decision_2_policy')
        logging.critical(f'*******************************************************************************')

    def make_decision_1(self, get_trains, revenue):
        if self.decision_1_policy == 'Available_Train_1':
            return self.check_available_train_1(get_trains)
        elif self.decision_1_policy == 'Available_Train_2':
            return self.check_available_train_2(get_trains)
        elif self.decision_1_policy == 'Available_Train_2_Or_Revenue':
            return self.check_available_train_2_or_revenue(get_trains, revenue)
        elif self.decision_1_policy == 'Available_Train_3':
            return self.check_available_train_3(get_trains)
        elif self.decision_1_policy == 'Available_Train_4':
            return self.check_available_train_4(get_trains)
        elif self.decision_1_policy == 'Available_Train_5':
            return self.check_available_train_5(get_trains)
        elif self.decision_1_policy == 'Accept_All':
            return self.accept_all()
        else:
            print("Invalid decision_1_policy")


    def make_decision_2(self, get_trains):
        if self.decision_2_policy == 'FCFS':
            return self.assign_FCFS(get_trains)
        elif self.decision_2_policy == 'Random':
            return self.assign_random(get_trains)
        else:
            print("Invalid decision_2_policy")

    def check_available_train_1(self, get_trains):
        if len(get_trains) >= 1:
            decision_1 = 1
        else:
            decision_1 = 0
        return decision_1
    
    def check_available_train_2(self, get_trains):
        if len(get_trains) >= 2:
            decision_1 = 1
        else:
            decision_1 = 0
        return decision_1
    
    def check_available_train_2_or_revenue(self, get_trains, revenue):
        if len(get_trains) >= 2 or revenue > 2*Policy.rejection_revenue:
            decision_1 = 1
        else:
            decision_1 = 0
        return decision_1
    
    def check_available_train_3(self, get_trains):
        if len(get_trains) >= 3:
            decision_1 = 1
        else:
            decision_1 = 0
        return decision_1
    
    def accept_all(self):
        decision_1 = 1
        return decision_1
    def check_available_train_4(self, get_trains):
        if len(get_trains) >= 4:
            decision_1 = 1
        else:
            decision_1 = 0
        return decision_1
    def check_available_train_5(self, get_trains):
        if len(get_trains) >= 5:
            decision_1 = 1
        else:
            decision_1 = 0
        return decision_1
    
    def assign_FCFS(self, get_trains):
        decision_2 = min(get_trains)
        return decision_2
    
    def assign_random(self, get_trains):
        decision_2 = np.random.choice(get_trains)
        return decision_2