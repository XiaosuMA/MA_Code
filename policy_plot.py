import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import savgol_filter

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
            for d_2 in Policy_Plot.decision_2_policy_list:
                avg_results = self.concat_avg_results(d_2)
                self.plot_losr_revenue_rejection_and_delay(avg_results, d_2)
                # self.plot_revenue_total(avg_results, d_2)
                # self.plot_imaginary_revenue_percentage(avg_results, d_2)
                # self.plot_reject_all_revenue_percentage(avg_results, d_2)
                # self.plot_delay_0_delivery_percentage(avg_results, d_2)
                # self.plot_delivery_percentage(avg_results, d_2)
                # self.plot_delay_true_waiting_percentage(avg_results, d_2)
                # self.plot_none_delay_percentage(avg_results, d_2)
                # self.plot_delay_nan_waiting_percentage(avg_results, d_2)
                # self.plot_failed_loading(avg_results, d_2)
                
        elif self.data_description == 'train_load':
            for d_2 in Policy_Plot.decision_2_policy_list:
                avg_results = self.concat_avg_results(d_2)
                # print(avg_results)
                self.plot_avg_total_passenger_extra(avg_results, d_2)
                self.plot_avg_train_load_percentage(avg_results, d_2)
                self.plot_avg_stu_onboard(avg_results, d_2)
        else:
            raise ValueError('Invalid data_description, please choose from "request" or "train_load"')
        

    def concat_avg_results(self, d_2):
        policy_list = []
        avg_results = pd.DataFrame()
        
        for d_1 in Policy_Plot.decision_1_policy_list:
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
        
    # Plot distribution of Delay_0_delivery (% Accepted), Delay_0_15_delivery, Delay_15_30_delivery, Delay_gt_30_delivery, Delay_0_waiting, Delay_nan_waiting(late_arrival), Delay_true_waiting
    def plot_delay_distribution(self, avg_results: pd.DataFrame):
        # Create subplots outside of the loop
    # Create subplots outside of the loop
        fig, axs = plt.subplots(len(avg_results)//2, 2, figsize=(10, 10), sharey=True)
        # Add main title
        fig.suptitle('Delay_Distribution_of_Policy', fontsize=12)
        # Define colors for even and odd subplots
        even_color = (244/255, 229/255, 170/255)  # Normalize RGB values (148/255, 189/255, 202/255)
        odd_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']

            delay_0_delivery = float(avg_results.loc[row,'Delay_0_delivery (of delivery)'].split(',')[1].strip())
            delay_0_15_delivery = float(avg_results.loc[row,'Delay_0_15_delivery'].split(',')[1].strip())
            delay_15_30_delivery = float(avg_results.loc[row,'Delay_15_30_delivery'].split(',')[1].strip())
            delay_gt_30_delivery = float(avg_results.loc[row,'Delay_gt_30_delivery'].split(',')[1].strip())
            none_delay_accepted = float(avg_results.loc[row,'None_Delay (of accepted)'].split(',')[1].strip())
            delay_0_waiting = float(avg_results.loc[row,'Delay_0_waiting (of accepted)'].split(',')[1].strip())
            delay_nan_waiting = float(avg_results.loc[row,'Delay_nan_waiting'].split(',')[1].strip())
            delay_true_waiting = float(avg_results.loc[row,'Delay_true_waiting'].split(',')[1].strip())
            data = [delay_0_delivery, delay_0_15_delivery, delay_15_30_delivery, delay_gt_30_delivery, none_delay_accepted, delay_0_waiting, delay_nan_waiting, delay_true_waiting]
            x_ticks = ['Delay_0_d', 'Delay_0_15_d', 'Delay_15_30_d', 'Delay_gt_30_d', 'None_Delay_A', 'Delay_0_w', 'Delay_nan_w', 'Delay_true_w']

            # Plot data on the subplot
            mycolor = even_color if row % 2 == 0 else odd_color
            ax = axs[row//2, row%2]
            ax.bar(x_ticks, data, label=policy, alpha= 1, color=mycolor)
            # ax.set_title(f'{policy}')
            # ax.set_xlabel('Delay Type')
            ax.set_ylabel('Log_Percentage')
            ax.legend()
            ax.set_xticks(range(len(x_ticks)))  # Set x-tick locations
            ax.set_xticklabels(x_ticks, rotation=60)  # Set x-tick labels
            ax.set_yscale('log')  # Set y-axis to logarithmic scale

            # Hide x-ticks for all but the last two subplots
            if row < len(avg_results) - 2:
                plt.setp(ax.get_xticklabels(), visible=False)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Delay_Distribution_each_Policy.png')
        plt.show()


    def plot_revenue_total(self, avg_results: pd.DataFrame, d_2: str):
        # plot the Revenue_Total for each policy
        x_ticks = []
        revenues = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            revenue_total = float(avg_results.loc[row,'STU_Total, Revenue_Total'].split(',')[1].strip())
            tick = policy
            x_ticks.append(tick)
            revenues.append(revenue_total)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks, revenues, label='Revenue Total', alpha=1, color=d2_color)
        for i, v in enumerate(revenues):
            plt.text(i, v + 25, "{:.2f}".format(v), ha='center', va='bottom')
            max_revenue = max(revenues)
        plt.axhline(y=max_revenue, color='black', alpha = 0.3, linestyle='--')
        # plt.annotate('Max: {:.2f}'.format(max_revenue), xy=(1, max_revenue), xytext=(8, 0), 
        #             xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.xticks(rotation=0, fontsize=14)
        plt.legend(fontsize=12 ,loc='upper right')
        # Set the limits of y-axis
        if self.passenger_demand_mode == 'constant':
            plt.ylim([1600, max(revenues) + 100])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([1600, max(revenues) + 100])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Revenue_Total_vs_Policy.png')
        plt.show()

    def plot_imaginary_revenue_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        imaginary_revenues = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            imaginary_revenue = float(avg_results.loc[row,'Imaginary_Revenue, PFA Ratio'].split(',')[1].strip())
            x_ticks.append(policy)
            imaginary_revenues.append(imaginary_revenue)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        bars = plt.bar(x_ticks, imaginary_revenues, label='Imaginary Revenue Ratios', alpha=1, color=d2_color)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, "{:.2f}%".format(yval*100), ha='center', va='bottom', fontsize = 12)
        # max_revenue = max(imaginary_revenues)
        # plt.axhline(y=max_revenue, color='black', alpha = 0.3, linestyle='--')
        # plt.annotate('Max: {:.2f}'.format(max_revenue), xy=(1, max_revenue), xytext=(8, 0), 
        #             xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.axhline(y=1.0, color='red', alpha = 0.3, linestyle='--')
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
            plt.ylim([0.4, 1.0])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([0.4, 1.0])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Imaginary_Revenue_vs_Policy.png')
        plt.show()


    def plot_reject_all_revenue_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            avg_reject_all_revenue_percentage_make = float(avg_results.loc[row,'Reject_All_Revenue, PFA Ratio'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(avg_reject_all_revenue_percentage_make)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        bars = plt.bar(x_ticks,y, label= 'All Rejection Ratios',  alpha=1, color=d2_color)
        for bar in bars:
            yval = bar.get_height()
            diff = yval - 1
            sign = '+' if diff >= 0 else '-'
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, sign + "{:.2f}%".format(abs(diff)*100), ha='center', va='bottom', fontsize=12) 
        plt.axhline(y=1.0, color='black', alpha = 0.3, linestyle='--')  # Add horizontal dashed line at y=1.0 
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.8)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.8)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Reject_All_Revenue_vs_Policy.png')
        plt.show()


    # Plot Delay_0_delivery (% Accepted)
    def plot_delay_0_delivery_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delay_0_delivery = float(avg_results.loc[row,'Delay_0_delivery (of delivery)'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delay_0_delivery)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks, y, label='On Time Ratio of Delivery', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.01, "{:.2f}%".format(v*100), ha='center', va='bottom', fontsize=12)

        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.8)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.8)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Delay_0_delivery_vs_Policy.png')
        plt.show()

    # Plot Delivery (% Total), we could integrate how many cargos into ÖPNV
    def plot_delivery_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delivery_percentage = float(avg_results.loc[row,'Delivered (of total)'].split(',')[1].strip()) + float(avg_results.loc[row,'On_Train'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delivery_percentage)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Delivery Ratio of Total', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', fontsize=12)

        plt.axhline(y=0.6, color='black', alpha = 0.3, linestyle='--')  # Add horizontal dashed line at y=0.6, 60% of total STU
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.0, top=1.0)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.0, top=1.0)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Delivery_Ratio_vs_Policy.png')
        plt.show()

    # Plot Delay_true_waiting, % to Accepted, How many worst case
    def plot_delay_true_waiting_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delay_true_waiting = float(avg_results.loc[row,'Delay_true_waiting'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delay_true_waiting)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Worst Cases to Accepted', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.001, "{:.2f}%".format(v*100), ha='center', va='bottom', fontsize=12)
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)
        if self.passenger_demand_mode == 'constant':
                plt.ylim(bottom=0.0, top=0.10)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.0, top=0.15)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Delay_true_waiting_accepted_vs_Policy.png')
        plt.show()

    def plot_none_delay_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            none_delay = float(avg_results.loc[row,'None_Delay (of accepted)'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(none_delay)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)
        plt.bar(x_ticks,y, label='Without Delay to Accepted', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.01, "{:.2f}%".format(v*100), ha='center', va='bottom', fontsize=12)
        plt.legend(fontsize=12 ,loc='upper left')
        plt.xticks(rotation=0, fontsize=14)
        plt.ylim(bottom=0.6)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_None_Delay_accepted_vs_Policy.png')
        plt.show()

    def plot_delay_nan_waiting_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delay_nan_waiting = float(avg_results.loc[row,'Delay_nan_waiting'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delay_nan_waiting)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)
        plt.bar(x_ticks,y, label='Delay Nan Waiting to Accepted', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.005, "{:.2f}%".format(v*100), ha='center', va='bottom', fontsize=12)
        plt.legend(fontsize=12 ,loc='upper left')
        plt.xticks(rotation=0, fontsize=14)
        plt.ylim(bottom=0.0, top=0.20)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Delay_nan_waiting_accepted_vs_Policy.png')
        plt.show()

    def plot_failed_loading(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            failed_loading = float(avg_results.loc[row,'Avg_Failed_Loading'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(failed_loading)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)
        plt.bar(x_ticks,y, label='Average Number of Failed Loading', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', fontsize=12)
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)
        plt.ylim(bottom=0.0, top=2.0)
        plt.yticks([]) 
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Failed_Loading_vs_Policy.png')
        plt.show()
    


    def plot_losr_revenue_rejection_and_delay(self, avg_results: pd.DataFrame, d_2: str):
        # line plot for 'Lost_Revenue_Rejection' and 'Lost_Revenue_Delay' from avg_results on one figure
        x_ticks = []
        losr_revenue_rejection = []
        losr_revenue_delay = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            losr_revenue_rejection.append(float(avg_results.loc[row,'Lost_Revenue_Rejection'].split(',')[1].strip()))
            losr_revenue_delay.append(float(avg_results.loc[row,'Lost_Revenue_Delay'].split(',')[1].strip()))
            x_ticks.append(policy)


        # Interpolate the data points
        x = np.arange(len(x_ticks))
        spl_rejection = make_interp_spline(x, losr_revenue_rejection, k=1)  # type: BSpline
        spl_delay = make_interp_spline(x, losr_revenue_delay, k=1)  # type: BSpline

        xnew = np.linspace(0, len(x_ticks)-1, num=1000, endpoint=True)

        # one one figure plot two lines
        plt.plot(xnew, spl_rejection(xnew), label='lost sales', alpha= 0.5, color='gray', linewidth=3)
        plt.plot(xnew, spl_delay(xnew), label='delay penalties', alpha = 1, color='black', linewidth=3)
        plt.axvline(x=2, color='gray', alpha = 0.5, linestyle='--')  # x=2 because Python uses 0-based indexing
        plt.title('Trade-off between Lost Sales and Delay Penalties')
        plt.legend(fontsize=12 ,loc='upper left')
        plt.xticks(x, x_ticks, rotation=0, fontsize=14)
        # plt.yticks([])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Lost_Revenue_Rejection_Delay_vs_Theta.png')
        plt.show()
            


############################################################################################################
# plot Total_Passenger_Extra for all Policy_abbr
    def plot_avg_total_passenger_extra(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            total_passenger_extra = float(avg_results.loc[row,'Total_Passenger_Extra'])
            x_ticks.append(policy)
            y.append(total_passenger_extra)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Total Extra Passenger', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', fontsize=14)
        # plt.title('Policy vs Total_Passenger_Extra')
        # plt.xlabel('Policy')  # Label for x-axis
        # plt.ylabel('Avg_Total_Passenger_Extra')  # Label for y-axis
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)  # Rotate x-axis labels
        plt.yticks([]) 
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Total_Passenger_Extra_vs_Policy.png')
        plt.show()       

# plot Average_Train_Load_Percentage for all Policy_abbr
    def plot_avg_train_load_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            avg_train_load_percentage = float(avg_results.loc[row,'Average_Train_Load_Percentage'])
            x_ticks.append(policy)
            y.append(avg_train_load_percentage)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Average Train Load', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.005, "{:.2f}%".format(v*100), ha='center', va='bottom', fontsize=14)
        # plt.title('Policy vs Average_Train_Load_Percentage')
        # plt.xlabel('Policy')  # Label for x-axis
        # plt.ylabel('Avg_Average_Train_Load_Percentage')  # Label for y-axis
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)  # Rotate x-axis labels
        plt.ylim(bottom=0.5)
        plt.yticks([]) 
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Average_Train_Load_Percentage_vs_Policy.png')
        plt.show()

# plot Average_STU_Onboard for all Policy_abbr
    def plot_avg_stu_onboard(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            avg_stu_onboard = float(avg_results.loc[row,'Average_STU_Onboard'])
            x_ticks.append(policy)
            y.append(avg_stu_onboard)
        if d_2 == 'Random':
            d2_color = (244/255, 229/255, 170/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Average Freight Onboard', alpha=1, color=d2_color)
        for i, v in enumerate(y):
            plt.text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom', fontsize=14)
        # plt.title('Policy vs Average_STU_Onboard')
        # plt.xlabel('Policy')  # Label for x-axis
        # plt.ylabel('Avg_Average_STU_Onboard')  # Label for y-axis
        plt.legend(fontsize=12 ,loc='upper right')
        plt.xticks(rotation=0, fontsize=14)
        plt.yticks([]) 
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Average_STU_Onboard_vs_Policy.png')
        plt.show()

############################################################################################################
# #Plotting for different passenger demand modes and data descriptions:

# plots = Policy_Plot(passenger_demand_mode='constant', data_description='request')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='linear', data_description='request')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='constant', data_description='train_load')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='linear', data_description='train_load')
# plots.plot_all()
