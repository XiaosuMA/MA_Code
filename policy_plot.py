import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
            avg_results_for_delay_distribution = self.concat_avg_for_delay_distribution()
            self.plot_delay_distribution(avg_results_for_delay_distribution)
            for d_2 in Policy_Plot.decision_2_policy_list:
                avg_results = self.concat_avg_results(d_2)
                self.plot_revenue_total(avg_results, d_2)
                self.plot_imaginary_revenue_percentage(avg_results, d_2)
                self.plot_reject_all_revenue_percentage(avg_results, d_2)
                self.plot_delay_0_delivery_percentage(avg_results, d_2)
                self.plot_delivery_percentage(avg_results, d_2)
                self.plot_delay_true_waiting_percentage(avg_results, d_2)
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
        fig, axs = plt.subplots(len(avg_results)//2, 2, figsize=(10, 10), sharey=True)
        # Add main title
        fig.suptitle('Delay_Distribution_of_Policy', fontsize=12)
        # Define colors for even and odd subplots
        even_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        odd_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']

            delay_0_delivery = float(avg_results.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
            delay_0_15_delivery = float(avg_results.loc[row,'Delay_0_15_delivery'].split(',')[1].strip())
            delay_15_30_delivery = float(avg_results.loc[row,'Delay_15_30_delivery'].split(',')[1].strip())
            delay_gt_30_delivery = float(avg_results.loc[row,'Delay_gt_30_delivery'].split(',')[1].strip())
            delay_0_waiting = float(avg_results.loc[row,'Delay_0_waiting'].split(',')[1].strip())
            delay_nan_waiting = float(avg_results.loc[row,'Delay_nan_waiting(late_arrival)'].split(',')[1].strip())
            delay_true_waiting = float(avg_results.loc[row,'Delay_true_waiting'].split(',')[1].strip())
            data = [delay_0_delivery, delay_0_15_delivery, delay_15_30_delivery, delay_gt_30_delivery, delay_0_waiting, delay_nan_waiting, delay_true_waiting]
            x_ticks = ['Delay_0_delivery', 'Delay_0_15_delivery', 'Delay_15_30_delivery', 'Delay_gt_30_delivery', 'Delay_0_waiting', 'Delay_nan_waiting', 'Delay_true_waiting']

            # Plot data on the subplot
            # Use the index of the loop to select a color
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
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks, revenues, label='policies', alpha=1, color=d2_color)
        max_revenue = max(revenues)
        plt.axhline(y=max_revenue, color='black', linestyle='--')
        plt.annotate('Max: {:.2f}'.format(max_revenue), xy=(1, max_revenue), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.xticks(rotation=0)
        plt.xlabel('Policy')
        plt.ylabel('Revenue_Total')
        plt.legend()
        plt.title('Revenue_Total for each policy')
        # Set the limits of y-axis
        if self.passenger_demand_mode == 'constant':
            plt.ylim([1600, max(revenues) + 100])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([1600, max(revenues) + 100])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Revenue_Total_vs_Policy.png')
        plt.show()

    def plot_imaginary_revenue_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        imaginary_revenues = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            imaginary_revenue = float(avg_results.loc[row,'Imaginary_Revenue, PRT to %'].split(',')[1].strip())
            x_ticks.append(policy)
            imaginary_revenues.append(imaginary_revenue)
        if d_2 == 'Random':
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        bars = plt.bar(x_ticks, imaginary_revenues, label='policies', alpha=1, color=d2_color)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), ha='center', va='bottom') # va: vertical alignment
        max_revenue = max(imaginary_revenues)
        plt.axhline(y=max_revenue, color='black', linestyle='--')
        # plt.annotate('Max: {:.2f}'.format(max_revenue), xy=(1, max_revenue), xytext=(8, 0), 
        #             xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.axhline(y=1.0, color='red', linestyle='--')
        # plt.annotate('1.0: get all imaginary revenue', xy=(1, 1.0), xytext=(8, 0),
        #             xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.title('Imaginary_Revenue, PRT to % for each policy')
        plt.xlabel('Policy')
        plt.ylabel('Imaginary_Revenue, PRT to %')
        plt.legend()
        plt.xticks(rotation=0)
        if self.passenger_demand_mode == 'constant':
            plt.ylim([0.0, 1.05])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Imaginary_Revenue_vs_Policy.png')
        plt.show()


    def plot_reject_all_revenue_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            avg_reject_all_revenue_percentage_make = float(avg_results.loc[row,'Reject_All_Revenue, PRT to %'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(avg_reject_all_revenue_percentage_make)
        if d_2 == 'Random':
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        bars = plt.bar(x_ticks,y, label= 'PRT to %',  alpha=1, color=d2_color)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, "+" + str(round(yval - 1 , 2)*100) + "%", ha='center', va='bottom') 

        plt.title('Reject_All_Revenue vs Policy Revenue, PRT to %')
        plt.axhline(y=1.0, color='black', linestyle='--')  # Add horizontal dashed line at y=1.0
        # plt.annotate('Reject All Revenue Line', xy=(1, 1.0), xytext=(8, 0), 
        #             xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Avg_Reject_All_Revenue, PRT to %')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.8)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.8)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Reject_All_Revenue_vs_Policy.png')
        plt.show()


    # Plot Delay_0_delivery (% Accepted)
    def plot_delay_0_delivery_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delay_0_delivery = float(avg_results.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delay_0_delivery)
        if d_2 == 'Random':
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Delay_0_delivery (% Accepted)', alpha=1, color=d2_color)
        plt.title('Policy vs Delay_0_delivery (% Accepted)')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Delay_0_delivery (% Accepted)')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.4)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.4)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Delay_0_delivery_vs_Policy.png')
        plt.show()

    # Plot Delivery (% Total), we could integrate how many cargos into ÖPNV
    def plot_delivery_percentage(self, avg_results: pd.DataFrame, d_2: str):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delivery_percentage = float(avg_results.loc[row,'Delivered (% Total)'].split(',')[1].strip()) + float(avg_results.loc[row,'On_Train'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delivery_percentage)
        if d_2 == 'Random':
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Delivery (% Total)', alpha=1, color=d2_color)
        plt.title('Policy vs Delivery (% Total)')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Delivery (% Total)')  # Label for y-axis
        plt.axhline(y=0.6, color='black', linestyle='--')  # Add horizontal dashed line at y=0.6, 60% of total STU
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.3)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.3)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Delivery_(% Total)_vs_Policy.png')
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
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Delay_true_waiting, % to Accepted', alpha=1, color=d2_color)
        plt.title('Policy vs Delay_true_waiting, % to Accepted')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Delay_true_waiting, % to Accepted')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Delay_true_waiting_vs_Policy.png')
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
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Total_Passenger_Extra', alpha=1, color=d2_color)
        plt.title('Policy vs Total_Passenger_Extra')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Avg_Total_Passenger_Extra')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
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
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Average_Train_Load_Percentage', alpha=1, color=d2_color)
        plt.title('Policy vs Average_Train_Load_Percentage')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Avg_Average_Train_Load_Percentage')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.ylim(bottom=0.675)
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
            d2_color = (148/255, 189/255, 202/255)  # Normalize RGB values
        elif d_2 == 'FCFS':
            d2_color = (196/255, 203/255, 229/255)  # Normalize RGB values
        plt.bar(x_ticks,y, label='Average_STU_Onboard', alpha=1, color=d2_color)
        plt.title('Policy vs Average_STU_Onboard')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Avg_Average_STU_Onboard')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.ylim(bottom=2.5)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\{d_2}_Average_STU_Onboard_vs_Policy.png')
        plt.show()

############################################################################################################
# plots = Policy_Plot(passenger_demand_mode='constant', data_description='request')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='linear', data_description='request')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='constant', data_description='train_load')
# plots.plot_all()

# plots = Policy_Plot(passenger_demand_mode='linear', data_description='train_load')
# plots.plot_all()
