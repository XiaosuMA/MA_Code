import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# function to format y ticks
def to_percent(y, position):
    return str(np.round(100 * y))
formatter = FuncFormatter(to_percent)


class Policy_Plot:

    # Policy:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 
                              #'Available_Train_2_Or_Revenue', 
                              'Available_Train_3', 'Available_Train_4', 'Available_Train_5']
    decision_2_policy_list = ['Random', 'FCFS']
    passenger_demand_mode_set = ['constant', 'linear']
    data_description_set = ['train_load', 'request']

    def __init__(self, passenger_demand_mode: str, data_description: str):
        self.passenger_demand_mode = passenger_demand_mode
        self.data_description = data_description

    def plot_all(self):
        if self.data_description == 'request':
            # avg_results_for_delay_distribution = self.concat_avg_for_delay_distribution()
            # self.plot_delay_distribution(avg_results_for_delay_distribution)
            avg_results = self.concat_avg_results()
            # self.plot_revenue_total(avg_results)
            self.plot_imaginary_revenue_percentage(avg_results)
            self.plot_reject_all_revenue_percentage(avg_results)
            self.plot_delay_0_delivery_percentage(avg_results)
            self.plot_none_delay_percentage(avg_results)
            self.plot_remaining_request_percentage(avg_results)
            self.plot_delay_true_accepted_percentage(avg_results)
        elif self.data_description == 'train_load':

            avg_results = self.concat_avg_results()
            # print(avg_results)
            self.plot_avg_total_passenger_extra(avg_results)
            self.plot_avg_train_load_percentage(avg_results)
        else:
            raise ValueError('Invalid data_description, please choose from "request" or "train_load"')
        

    def concat_avg_results(self):
        policy_list = []
        avg_results = pd.DataFrame()
        
        for d_1 in Policy_Plot.decision_1_policy_list:
            for d_2 in Policy_Plot.decision_2_policy_list:
                policy = f'{d_1}_{d_2}'
                policy_list.append(policy)
        print(policy_list)
        # Map 'A_T_1_R' to '\pi_1^{random}'
        policy_abbr_to_latex = {
            'A_A_R': r'$\pi_0^{random}$',
            'A_T_1_R': r'$\pi_1^{random}$',
            'A_T_2_R': r'$\pi_2^{random}$',
            'A_T_3_R': r'$\pi_3^{random}$',
            'A_T_4_R': r'$\pi_4^{random}$',
            'A_T_5_R': r'$\pi_5^{random}$',
            'A_A_F': r'$\pi_0^{FCFS}$',
            'A_T_1_F': r'$\pi_1^{FCFS}$',
            'A_T_2_F': r'$\pi_2^{FCFS}$',
            'A_T_3_F': r'$\pi_3^{FCFS}$',
            'A_T_4_F': r'$\pi_4^{FCFS}$',
            'A_T_5_F': r'$\pi_5^{FCFS}$'    
        }
        if self.data_description == 'request':
            for policy_item in policy_list:
                one_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\avg_results_{policy_item}.csv')
                avg_results = pd.concat([avg_results, one_result], ignore_index=True)

            avg_results.reset_index(drop = True, inplace = True)
            # Create abbreviation for Policies: for example, 'Available_Train_1_Random' -> 'A_T_1_R'
            for row in range(len(avg_results)):
                policy = avg_results.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                policy = '_'.join([word[0] for word in policy.split('_')])
                avg_results.loc[row,'Policy_abbr'] = policy_abbr_to_latex.get(policy, policy)
            return avg_results
        
        elif self.data_description == 'train_load':
            for policy_item in policy_list:
                one_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\avg_train_load_{policy_item}.csv')
                avg_results = pd.concat([avg_results, one_result], ignore_index=True)

            avg_results.reset_index(drop = True, inplace = True)
            # Create abbreviation for Policies: for example, 'Available_Train_1_Random' -> 'A_T_1_R'
            for row in range(len(avg_results)):
                policy = avg_results.loc[row,'Policy'].strip()
                policy = '_'.join([word[0] for word in policy.split('_')])
                avg_results.loc[row,'Policy_abbr'] = policy_abbr_to_latex.get(policy, policy)
            return avg_results
        else:
            raise ValueError('Invalid data_description, please choose from "request" or "train_load"')
        
    def concat_avg_for_delay_distribution(self):
        policy_list = []
        avg_results_for_delay_distribution = pd.DataFrame()
        for d_1 in Policy_Plot.decision_1_policy_list:
            for d_2 in Policy_Plot.decision_2_policy_list:
                policy = f'{d_1}_{d_2}'
                policy_list.append(policy)
        print(policy_list)
        if self.data_description == 'request':
            for policy_item in policy_list:
                one_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\avg_results_{policy_item}.csv')
                avg_results_for_delay_distribution = pd.concat([avg_results_for_delay_distribution, one_result], ignore_index=True)

            avg_results_for_delay_distribution.reset_index(drop = True, inplace = True)
            policy_abbr_to_latex = {
                'A_A_R': r'$\pi_0^{random}$',
                'A_T_1_R': r'$\pi_1^{random}$',
                'A_T_2_R': r'$\pi_2^{random}$',
                'A_T_3_R': r'$\pi_3^{random}$',
                'A_T_4_R': r'$\pi_4^{random}$',
                'A_T_5_R': r'$\pi_5^{random}$',
                'A_A_F': r'$\pi_0^{FCFS}$',
                'A_T_1_F': r'$\pi_1^{FCFS}$',
                'A_T_2_F': r'$\pi_2^{FCFS}$',
                'A_T_3_F': r'$\pi_3^{FCFS}$',
                'A_T_4_F': r'$\pi_4^{FCFS}$',
                'A_T_5_F': r'$\pi_5^{FCFS}$'    
            }
            for row in range(len(avg_results_for_delay_distribution)):
                policy = avg_results_for_delay_distribution.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                policy = '_'.join([word[0] for word in policy.split('_')])
                avg_results_for_delay_distribution.loc[row,'Policy_abbr'] = policy_abbr_to_latex.get(policy, policy)
            return avg_results_for_delay_distribution
        else:
            raise ValueError('Invalid data_description for delay distribution, must be "request"')

############################################################################################################

    def plot_revenue_total(self, avg_results: pd.DataFrame):
        # plot the Revenue_Total for each policy
        x_ticks = []
        revenues_random = []
        revenues_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            revenue_total_random = float(avg_results.loc[row,'STU_Total, Revenue_Total'].split(',')[1].strip())
            revenue_total_fcfs = float(avg_results.loc[row+1,'STU_Total, Revenue_Total'].split(',')[1].strip())
            tick = policy
            x_ticks.append(tick)
            revenues_random.append(revenue_total_random)
            revenues_fcfs.append(revenue_total_fcfs)
        print(x_ticks)
        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, revenues_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, revenues_fcfs, width, label='FCFS', alpha=1, color='gray')

        for i, v in enumerate(revenues_random):
            plt.text(i - width/2, v + 25, "{:.2f}".format(v), ha='center', va='bottom')
        for i, v in enumerate(revenues_fcfs):
            plt.text(i + width/2, v + 25, "{:.2f}".format(v), ha='center', va='bottom')

        max_revenue = max(max(revenues_random), max(revenues_fcfs))
        plt.axhline(y=max_revenue, color='black', alpha = 0.3, linestyle='--')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        plt.legend(fontsize=12 ,loc='upper right')
        # Set the limits of y-axis
        if self.passenger_demand_mode == 'constant':
            plt.ylim([1600, max_revenue + 100])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([1600, max_revenue + 100])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Revenue_Total_vs_Policy.png')
        plt.show()

    def plot_imaginary_revenue_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        imaginary_revenues_random = []
        imaginary_revenues_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            imag_revenue_random = float(avg_results.loc[row,'Imaginary_Revenue, PFA Ratio'].split(',')[1].strip())
            imag_revenue_fcfs = float(avg_results.loc[row+1,'Imaginary_Revenue, PFA Ratio'].split(',')[1].strip())
            x_ticks.append(policy)
            imaginary_revenues_random.append(imag_revenue_random)
            imaginary_revenues_fcfs.append(imag_revenue_fcfs)
        print(x_ticks)
        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, imaginary_revenues_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, imaginary_revenues_fcfs, width, label='FCFS', alpha=1, color='gray')

        # Display the residual value on top of each even bar
        for i in range(len(imaginary_revenues_fcfs)):
            residual = imaginary_revenues_fcfs[i] - imaginary_revenues_random[i]
            sign = '+' if residual >= 0 else '-'
            plt.text(i + width/2, imaginary_revenues_fcfs[i] + 0.01, sign + "{:.2f}%".format(abs(residual)*100), ha='center', va='bottom', fontsize=12)

        max_imaginary_revenue = max(max(imaginary_revenues_random), max(imaginary_revenues_fcfs))
        # plt.axhline(y=max_imaginary_revenue, color='red', alpha = 0.3, linestyle='--')
        plt.title('Imaginary Revenue Ratios')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        plt.legend(fontsize=12 ,loc='upper right')
        if self.passenger_demand_mode == 'constant':
            plt.ylim([0.4, max_imaginary_revenue + 0.1])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([0.4, max_imaginary_revenue + 0.1])
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Imaginary_Revenue_vs_Policy.png')
        plt.show()



    def plot_reject_all_revenue_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        reject_all_revenues_random = []
        reject_all_revenues_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            reject_all_random = float(avg_results.loc[row,'Reject_All_Revenue, PFA Ratio'].split(',')[1].strip())
            reject_all_fcfs = float(avg_results.loc[row+1,'Reject_All_Revenue, PFA Ratio'].split(',')[1].strip())
            x_ticks.append(policy)
            reject_all_revenues_random.append(reject_all_random)
            reject_all_revenues_fcfs.append(reject_all_fcfs)

        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, reject_all_revenues_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, reject_all_revenues_fcfs, width, label='FCFS', alpha=1, color='gray')

        # Display the residual value on top of each even bar
        for i in range(len(reject_all_revenues_fcfs)):
            residual = reject_all_revenues_fcfs[i] - reject_all_revenues_random[i]
            sign = '+' if residual >= 0 else '-'
            plt.text(i + width/2, reject_all_revenues_fcfs[i] + 0.01, sign + "{:.2f}%".format(abs(residual)*100), ha='center', va='bottom', fontsize=12)

        # plt.axhline(y=1.0, color='black', alpha = 0.3, linestyle='--')
        plt.legend(fontsize=12 ,loc='upper right')
        plt.title('All Rejection Revenue Ratios')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=1.0)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.8)
        # plt.yticks([])
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Reject_All_Revenue_vs_Policy.png')
        plt.show()


    def plot_delay_0_delivery_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        delay_0_delivery_random = []
        delay_0_delivery_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            delay_0_random = float(avg_results.loc[row,'Delay_0_delivery (of delivery)'].split(',')[1].strip())
            delay_0_fcfs = float(avg_results.loc[row+1,'Delay_0_delivery (of delivery)'].split(',')[1].strip())
            x_ticks.append(policy)
            delay_0_delivery_random.append(delay_0_random)
            delay_0_delivery_fcfs.append(delay_0_fcfs)

        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, delay_0_delivery_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, delay_0_delivery_fcfs, width, label='FCFS', alpha=1, color='gray')

        # Display the residual value on top of each even bar
        for i in range(len(delay_0_delivery_fcfs)):
            residual = delay_0_delivery_fcfs[i] - delay_0_delivery_random[i]
            sign = '+' if residual >= 0 else '-'
            plt.text(i + width/2, delay_0_delivery_fcfs[i] + 0.01, sign + "{:.2f}%".format(abs(residual)*100), ha='center', va='bottom', fontsize=12)

        plt.legend(fontsize=12 ,loc='upper left')
        plt.title('On Time Delivery Ratios')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.8, top = 1.07)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.8, top = 1.07)
        # plt.yticks([])
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Delay_0_delivery_vs_Policy.png')
        plt.show()

        
    def plot_none_delay_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        none_delay_random = []
        none_delay_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            none_random = float(avg_results.loc[row,'None_Delay (of accepted)'].split(',')[1].strip())
            none_fcfs = float(avg_results.loc[row+1,'None_Delay (of accepted)'].split(',')[1].strip())
            x_ticks.append(policy)
            none_delay_random.append(none_random)
            none_delay_fcfs.append(none_fcfs)

        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, none_delay_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, none_delay_fcfs, width, label='FCFS', alpha=1, color='gray')

        # Display the residual value on top of each odd bar
        for i in range(len(none_delay_fcfs)):
            residual = none_delay_fcfs[i] - none_delay_random[i]
            sign = '+' if residual >= 0 else '-'
            plt.text(i + width/2, none_delay_fcfs[i] + 0.01, sign + "{:.2f}%".format(abs(residual)*100), ha='center', va='bottom', fontsize=12)

        plt.legend(fontsize=12 ,loc='upper left')
        plt.title('None Delay Ratios')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.6)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.6)
        # plt.yticks([])
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\None_Delay_accepted_vs_Policy.png')
        plt.show()


    def plot_remaining_request_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        remaining_request_random = []
        remaining_request_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            delay_nan_random = float(avg_results.loc[row,'Delay_nan_waiting'].split(',')[1].strip())
            delay_0_waiting_random = float(avg_results.loc[row,'Delay_0_waiting (of accepted)'].split(',')[1].strip())
            remain_request_random = delay_nan_random + delay_0_waiting_random

            delay_nan_fcfs = float(avg_results.loc[row+1,'Delay_nan_waiting'].split(',')[1].strip())
            delay_0_waiting_fcfs = float(avg_results.loc[row+1,'Delay_0_waiting (of accepted)'].split(',')[1].strip())
            remain_request_fcfs = delay_nan_fcfs + delay_0_waiting_fcfs
            x_ticks.append(policy)
            remaining_request_random.append(remain_request_random)
            remaining_request_fcfs.append(remain_request_fcfs)

        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, remaining_request_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, remaining_request_fcfs, width, label='FCFS', alpha=1, color='gray')

        # Display the residual value on top of each odd bar
        for i in range(len(remaining_request_random)):
            residual = remaining_request_random[i] - remaining_request_fcfs[i]
            sign = '+' if residual >= 0 else '-'
            plt.text(i - width/2, remaining_request_random[i] + 0.005, sign + "{:.2f}%".format(abs(residual)*100), ha='center', va='bottom', fontsize=12)

        plt.legend(fontsize=11 ,loc='upper right')
        plt.title('Remaining Request Ratios')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        plt.ylim(bottom=0, top=0.21)
        #plt.yticks([])
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Delay_nan_waiting_accepted_vs_Policy.png')
        plt.show()

    # plot 'Delay_True (of accepted)' for random and FCFS policies
    def plot_delay_true_accepted_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        delay_true_accepted_random = []
        delay_true_accepted_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            delay_true_random = float(avg_results.loc[row,'Delay_True (of accepted)'].split(',')[1].strip())
            delay_true_fcfs = float(avg_results.loc[row+1,'Delay_True (of accepted)'].split(',')[1].strip())
            x_ticks.append(policy)
            delay_true_accepted_random.append(delay_true_random)
            delay_true_accepted_fcfs.append(delay_true_fcfs)

        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))
        width = 0.35

        plt.bar(x - width/2, delay_true_accepted_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, delay_true_accepted_fcfs, width, label='FCFS', alpha=1, color='gray')

        # Display the residual value on top of each odd bar
        for i in range(len(delay_true_accepted_random)):
            residual = delay_true_accepted_random[i] - delay_true_accepted_fcfs[i]
            sign = '+' if residual >= 0 else '-'
            plt.text(i - width/2, delay_true_accepted_random[i] + 0.01, sign + "{:.2f}%".format(abs(residual)*100), ha='center', va='bottom', fontsize=12)

        plt.legend(fontsize=12 ,loc='upper right')
        plt.title('Delay Ratios')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        plt.ylim(bottom=0.0, top=0.21)
        # plt.yticks([])
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Delay_True_of_accepted_vs_Policy.png')

############################################################################################################
# plot Total_Passenger_Extra for all Policy_abbr
    def plot_avg_total_passenger_extra(self, avg_results: pd.DataFrame):
        x_ticks = []
        total_passenger_extra_random = []
        total_passenger_extra_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            passenger_extra_random = float(avg_results.loc[row,'Total_Passenger_Extra'])
            passenger_extra_fcfs = float(avg_results.loc[row+1,'Total_Passenger_Extra'])
            x_ticks.append(policy)
            total_passenger_extra_random.append(passenger_extra_random)
            total_passenger_extra_fcfs.append(passenger_extra_fcfs)

        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, total_passenger_extra_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, total_passenger_extra_fcfs, width, label='FCFS', alpha=1, color='gray')

        for i, v in enumerate(total_passenger_extra_random):
            plt.text(i - width/2, v + 0.03, "{:.1f}".format(v), ha='center', va='bottom', fontsize = 12)
        for i, v in enumerate(total_passenger_extra_fcfs):
            plt.text(i + width/2, v + 0.03, "{:.1f}".format(v), ha='center', va='bottom', fontsize = 12)

        plt.legend(fontsize=12 ,loc='upper right')
        plt.title('Total Passenger Extra')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        plt.ylim(bottom=0.0)
        # plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Total_Passenger_Extra_vs_Policy.png')
        plt.show()   

# plot Average_Train_Load_Percentage for all Policy_abbr
    def plot_avg_train_load_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        avg_train_load_random = []
        avg_train_load_fcfs = []
        for row in range(0, len(avg_results), 2):
            policy = [avg_results.loc[row,'Policy_abbr'], avg_results.loc[row+1,'Policy_abbr']]
            train_load_random = float(avg_results.loc[row,'Average_Train_Load_Percentage'])
            train_load_fcfs = float(avg_results.loc[row+1,'Average_Train_Load_Percentage'])
            x_ticks.append(policy)
            avg_train_load_random.append(train_load_random)
            avg_train_load_fcfs.append(train_load_fcfs)

        x_ticks = [r'$\pi_0$', r'$\pi_1$', r'$\pi_2$', r'$\pi_3$', r'$\pi_4$', r'$\pi_5$']
        x = np.arange(len(x_ticks))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width/2, avg_train_load_random, width, label='Random', alpha=0.5, color='gray')
        plt.bar(x + width/2, avg_train_load_fcfs, width, label='FCFS', alpha=1, color='gray')

        for i, v in enumerate(avg_train_load_random):
            plt.text(i - width/2, v + 0.02, "{:.1f}%".format(v*100), ha='center', va='bottom', fontsize = 12)
        for i, v in enumerate(avg_train_load_fcfs):
            plt.text(i + width/2, v + 0.024, "{:.1f}%".format(v*100), ha='center', va='bottom', fontsize = 12)

        plt.legend(fontsize=12 ,loc='upper right')
        plt.title('Average Train Load Percentage')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        plt.ylim(bottom=0.5)
        # plt.yticks([])
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Trade_off_pics\Average_Train_Load_Percentage_vs_Policy.png')
        plt.show()



############################################################################################################
# # Plottings fdor all figs:

# plots = Policy_Plot(passenger_demand_mode='constant', data_description='request')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='linear', data_description='request')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='constant', data_description='train_load')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='linear', data_description='train_load')
# plots.plot_all()
