import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from class_simulator import Transport_Simulator
from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

policy_to_latex = {
    'Accept_All_FCFS': r'$\pi_0^{FCFS}$',
    'Available_Train_1_FCFS': r'$\pi_1^{FCFS}$',
    'Available_Train_2_FCFS': r'$\pi_2^{FCFS}$',
    'Available_Train_3_FCFS': r'$\pi_3^{FCFS}$',
    'Available_Train_4_FCFS': r'$\pi_4^{FCFS}$',
    'Available_Train_5_FCFS': r'$\pi_5^{FCFS}$'    
}

# Define a list of RGB colors
colors = [(250/255, 194/255, 197/255), (190/255, 230/255, 206/255), (196/255, 203/255, 229/255), (148/255, 189/255, 202/255), (161/255, 160/255, 166/255), (118/255, 98/255, 111/255), (62/255, 62/255, 64/255), (206/255, 197/255, 164/255)]
class Intensity_Plot:

    # Policy:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_3', 'Available_Train_4', 'Available_Train_5']# , 'Available_Train_2_Or_Revenue'
    decision_2_policy_list = ['Random', 'FCFS']
    passenger_demand_mode_set = ['constant', 'linear']
    data_description_set = ['train_load', 'request']
    cargo_arrival_intensity_set = Transport_Simulator.test_cargo_time_intensity_set 
    total_seeds = 50

    def __init__(self, passenger_demand_mode: str, data_description: str):
        self.passenger_demand_mode = passenger_demand_mode
        self.data_description = data_description

    def plot_all(self):
        if self.data_description == 'request':
            data_list = self.generate_data_list()
            # self.plot_delay_distruibution(data_list)
            self.plot_avg_revenue_total_with_intensity(data_list)
            self.plot_imaginary_revenue_percentage(data_list)
            self.plot_reject_all_percentage(data_list)
            self.plot_delay_0_delivery(data_list)
            self.plot_delivery_percentage(data_list)
            self.plot_delay_true_waiting(data_list)
            
        elif self.data_description == 'train_load':
            data_list = self.generate_data_list()
            self.plot_avg_total_passenger_extra(data_list)
            self.plot_avg_train_load_percentage(data_list)
            self.plot_avg_stu_onboard(data_list)

        else:
            print('Please specify the data_description as train_load or request')

    def generate_data_list(self):
        data_list = []
        if self.data_description == 'request':
            if self.passenger_demand_mode in Intensity_Plot.passenger_demand_mode_set:
                for intensity in Intensity_Plot.cargo_arrival_intensity_set:
                    avg_intensity_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/avg_results_intensity{intensity}.csv')
                    # Drop the rows where 'Seed_Time_Intensity, Policy' contains "Available_Train_2_Or_Revenue_FCFS"
                    avg_intensity_result = avg_intensity_result[~avg_intensity_result['Seed_Time_Intensity, Policy'].str.contains("Available_Train_2_Or_Revenue_FCFS")]
                    # Reset the index of the DataFrame
                    avg_intensity_result = avg_intensity_result.reset_index(drop=True)
                    data_list.append(avg_intensity_result)
            else:
                print('Please specify the passenger_demand_mode as constant or linear')
        elif self.data_description == 'train_load':
            if self.passenger_demand_mode in Intensity_Plot.passenger_demand_mode_set:
                for intensity in Intensity_Plot.cargo_arrival_intensity_set:
                    avg_intensity_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/avg_train_load_intensity{intensity}.csv')
                    # Drop the rows where 'Seed_Time_Intensity, Policy' contains "Available_Train_2_Or_Revenue_FCFS"
                    avg_intensity_result = avg_intensity_result[~avg_intensity_result['Policy'].str.contains("Available_Train_2_Or_Revenue_FCFS")]
                    # Reset the index of the DataFrame
                    avg_intensity_result = avg_intensity_result.reset_index(drop=True)
                    data_list.append(avg_intensity_result)
            else:
                print('Please specify the passenger_demand_mode as constant or linear')
        else:
            print('Please specify the data_description as train_load or request')
        return data_list
    
    ############################################################################################################################################################################
    
    def plot_delay_distruibution(self, data_list):
        # dataframes = [avg_intensity1, avg_intensity1p5, avg_intensity2, avg_intensity2p5, avg_intensity3, avg_intensity3p5]
        for row in range(len(data_list[0])):
            policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
            fig, axs = plt.subplots(int(np.round(len(data_list)/2, 0)), 2, figsize=(10, 10))  #sharey=True

            for i, df in enumerate(data_list):
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                delay_0_delivery = float(df.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
                delay_0_15_delivery = float(df.loc[row,'Delay_0_15_delivery'].split(',')[1].strip())
                delay_15_30_delivery = float(df.loc[row,'Delay_15_30_delivery'].split(',')[1].strip())
                delay_gt_30_delivery = float(df.loc[row,'Delay_gt_30_delivery'].split(',')[1].strip())
                delay_0_waiting = float(df.loc[row,'Delay_0_waiting'].split(',')[1].strip())
                delay_nan_waiting = float(df.loc[row,'Delay_nan_waiting(late_arrival)'].split(',')[1].strip())
                delay_true_waiting = float(df.loc[row,'Delay_true_waiting'].split(',')[1].strip())
                data = [delay_0_delivery, delay_0_15_delivery, delay_15_30_delivery, delay_gt_30_delivery, delay_0_waiting, delay_nan_waiting, delay_true_waiting]
                x_ticks = ['Delay_0_delivery', 'Delay_0_15_delivery', 'Delay_15_30_delivery', 'Delay_gt_30_delivery', 'Delay_0_waiting', 'Delay_nan_waiting', 'Delay_true_waiting']

                # Plot data on the i-th subplot
                ax = axs[i//2, i%2]
                ax.bar(x_ticks, data, label=intens, alpha=0.5)
                ax.set_title(f'{policy}')
                # ax.set_xlabel('Delay Type')
                ax.set_ylabel('Percentage')
                ax.legend()
                ax.set_xticks(range(len(x_ticks)))  # Set x-tick locations
                ax.set_xticklabels(x_ticks, rotation=60)  # Set x-tick labels
                ax.set_yscale('log')  # Set y-axis to logarithmic scale

                            # Hide x-ticks for all but the last two subplots
                if i < len(data_list) - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
            plt.tight_layout()
            plt.show()



    def plot_avg_revenue_total_with_intensity(self, data_list):
        y_max_2 = -np.inf  # Initialize the maximum y-value
        y_max_2p5 = -np.inf 
        for row in range(len(data_list[0])): # iterate over policies
            x = []
            y = []
            for df in data_list: # iterate over intensities
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_revenue = float(df.loc[row,'STU_Total, Revenue_Total'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_revenue)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)

            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))
            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color, linewidth=1.5) 
        plt.title('Intensity vs Average Revenue')
        # Draw the vertical lines at the maximum y-value
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Average Revenue')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_Revenue_Total_With_Intensity.png', dpi = 300)
        plt.show()
    
    def plot_imaginary_revenue_percentage(self, data_list):
        y_max_1p5 = -np.inf  # Initialize the maximum y-value
        y_max_2 = -np.inf  
        y_max_2p5 = -np.inf 
        # Plot Imaginary_Revenue, RRT to %
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_imaginary_revenue_percentage_make = float(df.loc[row,'Imaginary_Revenue, PRT to %'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_imaginary_revenue_percentage_make)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)

            y_max_1p5 = max(y_max_1p5, spl(1.5))
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))
            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color) 
        plt.title('Intensity vs Imaginary_Revenue, PRT to %')
        # plt.axhline(y=1.0, color='red', alpha = 0.3, linestyle='--')  # Add horizontal dashed line at y=1.0, where get 100% of imaginary revenue
        plt.axhline(y=0.8, color='red', alpha = 0.3, linestyle='--')  # Add horizontal dashed line at y=0.8, where get 80% of imaginary revenue
        plt.vlines(x=1.5, ymin=0, ymax=y_max_1p5, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Imaginary_Revenue, PRT to %')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.ylim(bottom=0.2)
        plt.tight_layout()  
        for num in plt.gca().get_yticklabels():
            if num.get_text() == '1.0':
                num.set_color('red')
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Imaginary_Revenue_PRT_to_%.png', dpi = 300)
        plt.show()

    def plot_reject_all_percentage(self, data_list):
        y_max_2 = -np.inf  # Initialize the maximum y-value
        y_max_2p5 = -np.inf
        # Reject_All_Revenue, % to RRT
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_reject_all_revenue_percentage_make = float(df.loc[row,'Reject_All_Revenue, PRT to %'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_reject_all_revenue_percentage_make)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))

            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color) 
        plt.title('Intensity vs Reject_All_Revenue, PRT to %')
        # plt.axhline(y=1.0, color='red',alpha = 0.3, linestyle='--')  # Add horizontal dashed line at y=1.0
        # plt.annotate('reject all profit', xy=(1, 1.0), xytext=(8, 0), 
        #             xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Avg_Reject_All_Revenue, PRT to %')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels

        # Set the limits of the y-axis to only show positive values
        plt.ylim(bottom=0.3)
        plt.tight_layout()
        # Change the color of the y-axis label at y=1.0 to red
        for num in plt.gca().get_yticklabels():
            if num.get_text() == '1.0':
                num.set_color('red')
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Reject_All_Revenue_PRT%.png', dpi = 300)
        plt.show()


    def plot_delay_0_delivery(self, data_list):
        y_max_2 = -np.inf  # Initialize the maximum y-value
        y_max_2p5 = -np.inf
        # Plot Delay_0_delivery (% Accepted)
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_delay_0_delivery_percentage = float(df.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_delay_0_delivery_percentage)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))
            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color) 
        plt.title('Intensity vs Delay_0_delivery (% Accepted)')
        plt.axhline(y=0.8, color='red', alpha = 0.3, linestyle='--')  
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Delay_0_delivery')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.ylim(bottom=0.5)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Delay_0_delivery_%_Accepted.png', dpi = 300)
        plt.show()

    def plot_delivery_percentage(self, data_list):
        y_max_2 = -np.inf
        y_max_2p5 = -np.inf
        # Plot Delivery (% Total), we could integrate how many cargos into ÖPNV
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_delivery_percentage = float(df.loc[row,'Delivered (% Total)'].split(',')[1].strip()) + float(df.loc[row,'On_Train'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_delivery_percentage)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))
            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color) 
        plt.title('Intensity vs Delivery (% Total)')
        plt.axhline(y=0.5, color='red', alpha = 0.3, linestyle='--')  # Add horizontal dashed line at y=0.5, more than half of cargos should be delivered by ÖPNV
        plt.annotate('50%', xy=(1, 0.5), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Delivery (% Total)')  # Label for y-axis
        plt.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left')    
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Delivery_%_Total.png', dpi = 300)
        plt.show()

    def plot_delay_true_waiting(self, data_list):
        y_max_2 = -np.inf
        y_max_2p5 = -np.inf
        # Plot Delay_true_waiting, % to Accepted, How many worst case
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_delay_true_waiting_percentage = float(df.loc[row,'Delay_true_waiting'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_delay_true_waiting_percentage)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))

            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color) 
        plt.title('Intensity vs Delay_true_waiting, % to Accepted')
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Delay_true_waiting, % to Accepted')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=0)  # Rotate x-axis labels
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Delay_true_waiting_%_to_Accepted.png', dpi = 300)
        plt.show()

############################################################################################################################################################################
        
    def plot_avg_total_passenger_extra(self, data_list):
        y_max_2 = -np.inf  # Initialize the maximum y-value
        y_max_2p5 = -np.inf
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Policy']
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity'][-3:])
                avg_total_passenger_extra = float(df.loc[row,'Total_Passenger_Extra'])
                x.append(intens)
                y.append(avg_total_passenger_extra)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=2)  # k = 3 or 5
            y_smooth = spl(xnew)
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))
            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0, color=color) 
        plt.title('Intensity vs Average Total Passenger Extra')
        # plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.axhline(y=10, color='red', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')
        plt.ylabel('Average Total Passenger Extra')
        plt.legend()
        plt.ylim(bottom=0)
        # Set y-axis to only show integer values
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        # Set y-axis to show ticks at multiples of 4
        plt.gca().yaxis.set_major_locator(MultipleLocator(base=4))
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_Total_Passenger_Extra.png', dpi = 300)
        plt.show()

    def plot_avg_train_load_percentage(self, data_list):
        y_max_2 = -np.inf  # Initialize the maximum y-value
        y_max_2p5 = -np.inf
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Policy']
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity'][-3:])
                avg_train_load_percentage = float(df.loc[row,'Average_Train_Load_Percentage'])
                x.append(intens)
                y.append(avg_train_load_percentage)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))

            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color) 
        plt.title('Intensity vs Average Train Load Percentage')
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.axhline(y=0.7, color='red', alpha = 0.3, linestyle='--')
        # plt.axhline(y=0.8, color='red', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')
        plt.ylabel('Average Train Load Percentage')
        plt.legend()# fontsize='small'
        plt.ylim(bottom=0.4)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_Train_Load_Percentage.png', dpi = 300)
        plt.show()
                
    def plot_avg_stu_onboard(self, data_list):
        y_max_2 = -np.inf  # Initialize the maximum y-value
        y_max_2p5 = -np.inf
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Policy']
                # Convert the policy name to its LaTeX representation
                if policy in policy_to_latex:
                    policy = policy_to_latex[policy]
                intens = float(df.loc[row,'Seed_Time_Intensity'][-3:])
                avg_stu_onboard = float(df.loc[row,'Average_STU_Onboard'])
                x.append(intens)
                y.append(avg_stu_onboard)

            # Use the index of the loop to select a color from the list
            color = colors[row % len(colors)]

            # Create a new set of x values for the spline
            xnew = np.linspace(min(x), max(x), 500)

            # Create a spline function
            spl = make_interp_spline(x, y, k=3)  # k = 3 or 5
            y_smooth = spl(xnew)
            y_max_2 = max(y_max_2, spl(2.0))
            y_max_2p5 = max(y_max_2p5, spl(2.5))
            plt.plot(xnew, y_smooth, label=policy, alpha = 1.0 , color=color) 
        plt.title('Intensity vs Average STU Onboard')
        plt.vlines(x=2.0, ymin=0, ymax=y_max_2, color='black', alpha = 0.3, linestyle='--')
        plt.vlines(x=2.5, ymin=0, ymax=y_max_2p5, color='black', alpha = 0.3, linestyle='--')
        plt.xlabel('Intensity')
        plt.ylabel('Average STU Onboard')
        plt.legend()
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Outputs\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_STU_Onboard.png', dpi = 300)
        plt.show()


# check_run = Intensity_Plot(passenger_demand_mode = 'constant', data_description = 'request')
# check_run.plot_all()

# check_run = Intensity_Plot(passenger_demand_mode = 'linear', data_description = 'request')
# check_run.plot_all()

# check_run = Intensity_Plot(passenger_demand_mode = 'constant', data_description = 'train_load')
# check_run.plot_all()

# check_run = Intensity_Plot(passenger_demand_mode = 'linear', data_description = 'train_load')
# check_run.plot_all()

