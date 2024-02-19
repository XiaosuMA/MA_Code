import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from class_simulator import Transport_Simulator

class Intensity_Plot:

    # Policy:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 'Available_Train_2_Or_Revenue', 'Available_Train_3']
    decision_2_policy_list = ['Random', 'FCFS']
    passenger_demand_mode_set = ['constant', 'linear']
    data_description_set = ['train_load', 'request']
    cargo_arrival_intensity_set = Transport_Simulator.test_cargo_time_intensity_set 
    total_seeds = 20

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
                    avg_intensity_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/avg_results_intensity{intensity}.csv')
                    data_list.append(avg_intensity_result)
            else:
                print('Please specify the passenger_demand_mode as constant or linear')
        elif self.data_description == 'train_load':
            if self.passenger_demand_mode in Intensity_Plot.passenger_demand_mode_set:
                for intensity in Intensity_Plot.cargo_arrival_intensity_set:
                    avg_intensity_result = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/avg_train_load_intensity{intensity}.csv')
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
        for row in range(len(data_list[0])): # iterate over policies
            x = []
            y = []
            for df in data_list: # iterate over intensities
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_revenue = float(df.loc[row,'STU_Total, Revenue_Total'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_revenue)

            plt.plot(x,y, label= policy) #+ '_' + 'Avg_Revenue_Total_With_Intensity'
        plt.title('Intensity vs Average Revenue')
        plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Average Revenue')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=90)  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_Revenue_Total_With_Intensity.png', dpi = 300)
        plt.show()
    
    def plot_imaginary_revenue_percentage(self, data_list):
        # Plot Imaginary_Revenue, RRT to %
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_imaginary_revenue_percentage_make = float(df.loc[row,'Imaginary_Revenue, PRT to %'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_imaginary_revenue_percentage_make)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Imaginary_Revenue, PRT to %')
        plt.axhline(y=1.0, color='black', linestyle='--')  # Add horizontal dashed line at y=1.0, where get 100% of imaginary revenue
        plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Imaginary_Revenue, PRT to %')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=90)  # Rotate x-axis labels
        plt.tight_layout()  
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Imaginary_Revenue_PRT_to_%.png', dpi = 300)
        plt.show()

    def plot_reject_all_percentage(self, data_list):
        # Reject_All_Revenue, % to RRT
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_reject_all_revenue_percentage_make = float(df.loc[row,'Reject_All_Revenue, PRT to %'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_reject_all_revenue_percentage_make)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Reject_All_Revenue, PRT to %')
        plt.axhline(y=1.0, color='black', linestyle='--')  # Add horizontal dashed line at y=1.0
        plt.annotate('Reject All Revenue Line', xy=(1, 1.0), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Avg_Reject_All_Revenue, PRT to %')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=90)  # Rotate x-axis labels

        # Set the limits of the y-axis to only show positive values
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Reject_All_Revenue_PRT%.png', dpi = 300)
        plt.show()


    def plot_delay_0_delivery(self, data_list):
        # Plot Delay_0_delivery (% Accepted)
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_delay_0_delivery_percentage = float(df.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_delay_0_delivery_percentage)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Delay_0_delivery (% Accepted)')
        plt.axhline(y=0.8, color='black', linestyle='--')  # Add horizontal dashed line at y=0.8, service level agreement
        plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Delay_0_delivery')  # Label for y-axis   
        plt.legend()
        plt.xticks(rotation=90)  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Delay_0_delivery_%_Accepted.png', dpi = 300)
        plt.show()

    def plot_delivery_percentage(self, data_list):
        # Plot Delivery (% Total), we could integrate how many cargos into ÖPNV
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_delivery_percentage = float(df.loc[row,'Delivered (% Total)'].split(',')[1].strip()) + float(df.loc[row,'On_Train'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_delivery_percentage)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Delivery (% Total)')
        plt.axhline(y=0.6, color='black', linestyle='--')  # Add horizontal dashed line at y=0.6, more than half of cargos should be delivered by ÖPNV
        plt.annotate('At least delivery 60%', xy=(1, 0.6), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Delivery (% Total)')  # Label for y-axis
        plt.legend()   
        plt.xticks(rotation=90)  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Delivery_%_Total.png', dpi = 300)
        plt.show()

    def plot_delay_true_waiting(self, data_list):
        # Plot Delay_true_waiting, % to Accepted, How many worst case
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
                intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
                avg_delay_true_waiting_percentage = float(df.loc[row,'Delay_true_waiting'].split(',')[1].strip())
                x.append(intens)
                y.append(avg_delay_true_waiting_percentage)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Delay_true_waiting, % to Accepted')
        plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
        plt.axvline(x=2.0, color='black', linestyle='--')
        plt.xlabel('Intensity')  # Label for x-axis
        plt.ylabel('Delay_true_waiting, % to Accepted')  # Label for y-axis
        plt.legend()
        plt.xticks(rotation=90)  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Delay_true_waiting_%_to_Accepted.png', dpi = 300)
        plt.show()

############################################################################################################################################################################
        
    def plot_avg_total_passenger_extra(self, data_list):
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Policy']
                intens = float(df.loc[row,'Seed_Time_Intensity'][-3:])
                avg_total_passenger_extra = float(df.loc[row,'Total_Passenger_Extra'])
                x.append(intens)
                y.append(avg_total_passenger_extra)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Average Total Passenger Extra')
        plt.axvline(x=2.0, color='black', linestyle='--')
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')
        plt.ylabel('Average Total Passenger Extra')
        plt.legend()
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_Total_Passenger_Extra.png', dpi = 300)
        plt.show()

    def plot_avg_train_load_percentage(self, data_list):
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Policy']
                intens = float(df.loc[row,'Seed_Time_Intensity'][-3:])
                avg_train_load_percentage = float(df.loc[row,'Average_Train_Load_Percentage'])
                x.append(intens)
                y.append(avg_train_load_percentage)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Average Train Load Percentage')
        plt.axvline(x=2.0, color='black', linestyle='--')
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')
        plt.ylabel('Average Train Load Percentage')
        plt.legend()
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_Train_Load_Percentage.png', dpi = 300)
        plt.show()
                
    def plot_avg_stu_onboard(self, data_list):
        for row in range(len(data_list[0])):
            x = []
            y = []
            for df in data_list:
                policy = data_list[0].loc[row,'Policy']
                intens = float(df.loc[row,'Seed_Time_Intensity'][-3:])
                avg_stu_onboard = float(df.loc[row,'Average_STU_Onboard'])
                x.append(intens)
                y.append(avg_stu_onboard)

            plt.plot(x,y, label= policy) 
        plt.title('Intensity vs Average STU Onboard')
        plt.axvline(x=2.0, color='black', linestyle='--')
        plt.axvline(x=2.5, color='black', linestyle='--')
        plt.xlabel('Intensity')
        plt.ylabel('Average STU Onboard')
        plt.legend()
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_{self.passenger_demand_mode}/Intensity_vs_Avg_STU_Onboard.png', dpi = 300)
        plt.show()


check_run = Intensity_Plot(passenger_demand_mode = 'constant', data_description = 'train_load')
check_run.plot_all()