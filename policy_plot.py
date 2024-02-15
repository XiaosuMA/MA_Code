import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Policy_Plot:

    # Policy:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']
    decision_2_policy_list = ['Random', 'FCFS']

    def __init__(self, passenger_demand_mode: str):
        self.passenger_demand_mode = passenger_demand_mode

    def plot_all(self):
        avg_results = self.concat_avg_results()
        self.plot_delay_distribution(avg_results)
        self.plot_revenue_total(avg_results)
        self.plot_imaginary_revenue_percentage(avg_results)
        self.plot_reject_all_revenue_percentage(avg_results)
        self.plot_delay_0_delivery_percentage(avg_results)
        self.plot_delivery_percentage(avg_results)
        self.plot_delay_true_waiting_percentage(avg_results)

    def concat_avg_results(self):
        policy_list = []
        avg_results = pd.DataFrame()
        for d_1 in Policy_Plot.decision_1_policy_list:
            for d_2 in Policy_Plot.decision_2_policy_list:
                policy = f'{d_1}_{d_2}'
                policy_list.append(policy)
        print(policy_list)
        
        for policy_item in policy_list:
            one_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\avg_results_{policy_item}.csv')
            avg_results = pd.concat([avg_results, one_result], ignore_index=True)

        avg_results.reset_index(drop = True, inplace = True)


        # Create abbreviation for Policies: for example, 'Available_Train_1_Random' -> 'A_T_1_R'
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
            policy = '_'.join([word[0] for word in policy.split('_')])
            avg_results.loc[row,'Policy_abbr'] = policy
        return avg_results


    # Plot distribution of Delay_0_delivery (% Accepted), Delay_0_15_delivery, Delay_15_30_delivery, Delay_gt_30_delivery, Delay_0_waiting, Delay_nan_waiting(late_arrival), Delay_true_waiting
    def plot_delay_distribution(self, avg_results: pd.DataFrame):
        # Create subplots outside of the loop
        fig, axs = plt.subplots(len(avg_results)//2, 2, figsize=(10, 10), sharey=True)
        # Add main title
        fig.suptitle('Delay_Distribution_of_Policy', fontsize=12)

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
            ax = axs[row//2, row%2]
            ax.bar(x_ticks, data, label=policy, alpha=0.5)
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


    def plot_revenue_total(self, avg_results: pd.DataFrame):
        # plot the Revenue_Total for each policy
        x_ticks = []
        revenues = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            revenue_total = float(avg_results.loc[row,'STU_Total, Revenue_Total'].split(',')[1].strip())
            tick = policy
            x_ticks.append(tick)
            revenues.append(revenue_total)
        plt.bar(x_ticks, revenues, label='policies', alpha=0.5)
        max_revenue = max(revenues)
        plt.axhline(y=max_revenue, color='black', linestyle='--')
        plt.annotate('Max: {:.2f}'.format(max_revenue), xy=(1, max_revenue), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.xticks(rotation=60)
        plt.xlabel('Policy')
        plt.ylabel('Revenue_Total')
        plt.legend()
        plt.title('Revenue_Total for each policy')
        # Set the limits of y-axis
        if self.passenger_demand_mode == 'constant':
            plt.ylim([1600, max(revenues) + 100])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([600, max(revenues) + 100])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Revenue_Total_vs_Policy.png')
        plt.show()

# plot Imaginary_Revenue, PRT to % for each policy
    def plot_imaginary_revenue_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        imaginary_revenues = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            imaginary_revenue = float(avg_results.loc[row,'Imaginary_Revenue, PRT to %'].split(',')[1].strip())
            x_ticks.append(policy)
            imaginary_revenues.append(imaginary_revenue)
        plt.bar(x_ticks, imaginary_revenues, label='policies', alpha=0.5)
        max_revenue = max(imaginary_revenues)
        plt.axhline(y=max_revenue, color='black', linestyle='--')
        plt.annotate('Max: {:.2f}'.format(max_revenue), xy=(1, max_revenue), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.axhline(y=1.0, color='red', linestyle='--')
        plt.annotate('1.0: get all imaginary revenue', xy=(1, 1.0), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.title('Imaginary_Revenue, PRT to % for each policy')
        plt.xlabel('Policy')
        plt.ylabel('Imaginary_Revenue, PRT to %')
        plt.legend()
        plt.xticks(rotation=60)
        if self.passenger_demand_mode == 'constant':
            plt.ylim([0.5, 1.1])
        elif self.passenger_demand_mode == 'linear':
            plt.ylim([0.2, 1.1])
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Imaginary_Revenue_vs_Policy.png')
        plt.show()


    # Reject_All_Revenue, PRT to %
    def plot_reject_all_revenue_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            avg_reject_all_revenue_percentage_make = float(avg_results.loc[row,'Reject_All_Revenue, PRT to %'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(avg_reject_all_revenue_percentage_make)

        plt.bar(x_ticks,y, label= 'PRT to %',  alpha=0.5)
        plt.title('Reject_All_Revenue vs Policy Revenue, PRT to %')
        plt.axhline(y=1.0, color='black', linestyle='--')  # Add horizontal dashed line at y=1.0
        plt.annotate('Reject All Revenue Line', xy=(1, 1.0), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Avg_Reject_All_Revenue, PRT to %')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=60)  # Rotate x-axis labels
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.4)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.4)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Reject_All_Revenue_vs_Policy.png')
        plt.show()


    # Plot Delay_0_delivery (% Accepted)
    def plot_delay_0_delivery_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delay_0_delivery = float(avg_results.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delay_0_delivery)

        plt.bar(x_ticks,y, label='Delay_0_delivery (% Accepted)', alpha=0.5)
        plt.title('Policy vs Delay_0_delivery (% Accepted)')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Delay_0_delivery (% Accepted)')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=60)  # Rotate x-axis labels
        if self.passenger_demand_mode == 'constant':
            plt.ylim(bottom=0.4)
        elif self.passenger_demand_mode == 'linear':
            plt.ylim(bottom=0.4)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Delay_0_delivery_vs_Policy.png')
        plt.show()

    # Plot Delivery (% Total), we could integrate how many cargos into ÖPNV
    def plot_delivery_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delivery_percentage = float(avg_results.loc[row,'Delivered (% Total)'].split(',')[1].strip()) + float(avg_results.loc[row,'On_Train'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delivery_percentage)

        plt.bar(x_ticks,y, label='Delivery (% Total)', alpha=0.5)
        plt.title('Policy vs Delivery (% Total)')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Delivery (% Total)')  # Label for y-axis
        plt.axhline(y=0.6, color='black', linestyle='--')  # Add horizontal dashed line at y=0.6, 60% of total STU
        plt.legend()
        plt.xticks(rotation=60)  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Delivery_(% Total)_vs_Policy.png')
        plt.show()

    # Plot Delay_true_waiting, % to Accepted, How many worst case
    def plot_delay_true_waiting_percentage(self, avg_results: pd.DataFrame):
        x_ticks = []
        y = []
        for row in range(len(avg_results)):
            policy = avg_results.loc[row,'Policy_abbr']
            delay_true_waiting = float(avg_results.loc[row,'Delay_true_waiting'].split(',')[1].strip())
            x_ticks.append(policy)
            y.append(delay_true_waiting)

        plt.bar(x_ticks,y, label='Delay_true_waiting, % to Accepted', alpha=0.5)
        plt.title('Policy vs Delay_true_waiting, % to Accepted')
        plt.xlabel('Policy')  # Label for x-axis
        plt.ylabel('Delay_true_waiting, % to Accepted')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=60)  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Policy_Selection_Outputs\Passenger_{self.passenger_demand_mode}\Delay_true_waiting_vs_Policy.png')
        plt.show()


plots = Policy_Plot('linear')
plots.plot_all()

