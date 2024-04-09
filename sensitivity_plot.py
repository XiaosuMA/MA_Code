import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define a function to format y ticks
def to_percent(y, position):
    return str(np.round(100 * y))
formatter = FuncFormatter(to_percent)

from class_simulator import Transport_Simulator
class Case_Plot:

    # Policy:
    decision_1_policy_list = ['Accept_All', 'Available_Train_1', 'Available_Train_2', 
                              #'Available_Train_2_Or_Revenue', 
                              'Available_Train_3', 'Available_Train_4', 'Available_Train_5']
    decision_2_policy_list = ['Random', 'FCFS']
    passenger_demand_mode_set = ['constant', 'linear']
    data_description_set = ['train_load', 'request']
    sensitivity_pattern_set = Transport_Simulator.sensitivity_pattern_set
    #['Passenger_Demand_Time_Intensity', 'STU_Demand_Station_Intensity', 'STU_Demand_Time_Intensity'] 
    STU_arrival_over_station_set = Transport_Simulator.STU_arrival_over_station_set
    # ['uniform', 'hermes_peaks']   



    def plot_all(self):
        avg_results = self.concat_avg_results()
        self.plot_revenue_total(avg_results)
        self.plot_imaginary_revenue_percentage(avg_results)
        self.plot_reject_all_revenue_percentage(avg_results)
        self.plot_delivery_ratio(avg_results)
        self.plot_delay_0_delivery_percentage(avg_results)
        self.plot_none_delay_percentage(avg_results)
        self.plot_remaining_request_percentage(avg_results)
        self.plot_delay_true_accepted_percentage(avg_results)
        self.plot_avg_total_passenger_extra(avg_results)
        self.plot_avg_train_load_percentage(avg_results)

        

    def concat_avg_results(self):
        avg_results = pd.DataFrame()
        basic_stu_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Passenger_Demand_Time_Intensity_Sensitivity\avg_results_Passenger_constant.csv')
        basic_load_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Passenger_Demand_Time_Intensity_Sensitivity\avg_train_load_Passenger_constant.csv')

        basic_data = pd.concat([basic_stu_data, basic_load_data], axis=1)
        basic_data.drop(columns=['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode'], inplace=True)
        # print(pa_constant_data)


        pa_linear_stu_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Passenger_Demand_Time_Intensity_Sensitivity\avg_results_Passenger_linear.csv')
        pa_linear_load_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Passenger_Demand_Time_Intensity_Sensitivity\avg_train_load_Passenger_linear.csv')
        station_hermes_stu_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\STU_Demand_Station_Intensity_Sensitivity\avg_results_Station_hermes_peaks.csv')
        station_hermes_load_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\STU_Demand_Station_Intensity_Sensitivity\avg_train_load_Station_hermes_peaks.csv')
        mixed_stu_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\STU_Demand_Station_Intensity_Sensitivity\avg_results_Mixed.csv')
        mixed_load_data = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\STU_Demand_Station_Intensity_Sensitivity\avg_train_load_Mixed.csv')

        pa_linear_data = pd.concat([pa_linear_stu_data, pa_linear_load_data], axis=1)
        pa_linear_data.drop(columns=['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode'], inplace=True)
        # print(pa_linear_data)
        station_hermes_data = pd.concat([station_hermes_stu_data, station_hermes_load_data], axis=1)
        station_hermes_data.drop(columns=['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode'], inplace=True)
        # print(station_hermes_data)
        mixed_data = pd.concat([mixed_stu_data, mixed_load_data], axis=1)
        mixed_data.drop(columns=['Seed_Time_Intensity, Policy', 'Passenger_Demand_Mode, STU_Demand_Mode'], inplace=True)

        case_data = pd.concat([basic_data, pa_linear_data, station_hermes_data, mixed_data], axis=0)
        cols = case_data.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Passenger_Demand_Mode')))
        cols.insert(1, cols.pop(cols.index('STU_Demand_Mode')))
        case_data = case_data.reindex(columns=cols)

        avg_results = case_data
        avg_results = case_data.reset_index(drop=True)
        avg_results.to_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\summary_avg_results.csv', index=False)

        return avg_results
    
############################################################################################################

    def plot_revenue_total(self, avg_results: pd.DataFrame):
        revenue_total = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2]  
        for row in avg_results.index:
            revenue = float(avg_results.loc[row, 'STU_Total, Revenue_Total'].split(',')[1].strip())
            revenue_total.append(revenue)
        
        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], revenue_total[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(revenue_total):
            plt.text(i, v + 20, "{:.2f}".format(v), ha='center', va='bottom')
        plt.title('Total Revenue')
        plt.xlabel('Cases')
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Total_Revenue.png', dpi = 300)
        plt.show()



    def plot_imaginary_revenue_percentage(self, avg_results: pd.DataFrame):
        imag_revenue_percentage = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            imag_revenue_str = avg_results.loc[row, 'Imaginary_Revenue, PFA Ratio']
            imag_revenue = float(imag_revenue_str.split(',')[1].strip())
            imag_revenue_percentage.append(imag_revenue)

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], imag_revenue_percentage[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(imag_revenue_percentage):
            plt.text(i, v + 0.01, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('Imaginary Revenue Percentage')
        plt.xlabel('Cases')
        plt.ylabel('Percentage')
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Imaginary_Revenue.png', dpi = 300)
        plt.show()


    def plot_reject_all_revenue_percentage(self, avg_results: pd.DataFrame):
        reject_all_revenue_percentage = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2]  
        for row in avg_results.index:
            reject_all_revenue_str = avg_results.loc[row, 'Reject_All_Revenue, PFA Ratio']
            reject_all_revenue = float(reject_all_revenue_str.split(',')[1].strip())
            reject_all_revenue_percentage.append(reject_all_revenue)

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], reject_all_revenue_percentage[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(reject_all_revenue_percentage):
            plt.text(i, v + 0.01, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('Reject All Revenue Percentage')
        plt.xlabel('Cases')
        plt.ylabel('Percentage')
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Reject_All_Revenue.png', dpi = 300)
        plt.show()


    def plot_delivery_ratio(self, avg_results: pd.DataFrame):
        delivery_ratio = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            delivery_percentage = float(avg_results.loc[row,'Delivered (of total)'].split(',')[1].strip()) + float(avg_results.loc[row,'On_Train'].split(',')[1].strip())
            delivery_ratio.append(float(delivery_percentage))

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], delivery_ratio[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(delivery_ratio):
            plt.text(i, v + 0.01, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('Delivery Ratio')
        plt.xlabel('Cases')
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Delivery_Ratio.png', dpi = 300)
        plt.show()


    def plot_delay_0_delivery_percentage(self, avg_results: pd.DataFrame):
        delay_0_delivery_percentage = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            delay_0_delivery_str = avg_results.loc[row, 'Delay_0_delivery (of delivery)']
            delay_0_delivery = float(delay_0_delivery_str.split(',')[1].strip())
            delay_0_delivery_percentage.append(delay_0_delivery)

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], delay_0_delivery_percentage[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(delay_0_delivery_percentage):
            plt.text(i, v + 0.01, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('Delay 0 Delivery Percentage')
        plt.xlabel('Cases')
        plt.ylabel('Percentage')
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Delay_0_Delivery.png', dpi = 300)
        plt.show()

        
    def plot_none_delay_percentage(self, avg_results: pd.DataFrame):
        none_delay_percentage = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            none_delay_str = avg_results.loc[row, 'None_Delay (of accepted)']
            none_delay = float(none_delay_str.split(',')[1].strip())
            none_delay_percentage.append(none_delay)

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], none_delay_percentage[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(none_delay_percentage):
            plt.text(i, v + 0.01, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('None Delay Percentage')
        plt.xlabel('Cases')
        plt.ylabel('Percentage')
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\None_Delay.png', dpi = 300)
        plt.show()


    def plot_remaining_request_percentage(self, avg_results: pd.DataFrame):
        remaining_request_percentage = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            delay_0_waiting_str = avg_results.loc[row, 'Delay_0_waiting (of accepted)']
            delay_nan_waiting_str = avg_results.loc[row, 'Delay_nan_waiting']
            delay_0_waiting = float(delay_0_waiting_str.split(',')[1].strip())
            delay_nan_waiting = float(delay_nan_waiting_str.split(',')[1].strip())
            remaining_request = delay_0_waiting + delay_nan_waiting
            remaining_request_percentage.append(remaining_request)

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], remaining_request_percentage[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(remaining_request_percentage):
            plt.text(i, v + 0.001, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('Remaining Request Percentage')
        plt.xlabel('Cases')
        plt.ylabel('Percentage')
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Remaining_Request.png', dpi = 300)
        plt.show()


    # plot 'Delay_True (of accepted)' for random and FCFS policies
    def plot_delay_true_accepted_percentage(self, avg_results: pd.DataFrame):
        delay_true_accepted_percentage = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            delay_true_accepted_str = avg_results.loc[row, 'Delay_True (of accepted)']
            delay_true_accepted = float(delay_true_accepted_str.split(',')[1].strip())
            delay_true_accepted_percentage.append(delay_true_accepted)

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], delay_true_accepted_percentage[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(delay_true_accepted_percentage):
            plt.text(i, v + 0.0001, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('Delay True Accepted Percentage')
        plt.xlabel('Cases')
        plt.ylabel('Percentage')
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Delay_True_Accepted.png', dpi = 300)
        plt.show()



# ############################################################################################################
    def plot_avg_total_passenger_extra(self, avg_results: pd.DataFrame):
        avg_total_passenger_extra = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            avg_total_passenger_extra_str = avg_results.loc[row, 'Total_Passenger_Extra']
            avg_total_passenger_extra.append(float(avg_total_passenger_extra_str))

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], avg_total_passenger_extra[i], color='gray', alpha=alphas[i])
        
        for i, v in enumerate(avg_total_passenger_extra):
            plt.text(i, v + 0.01, "{:.2f}".format(v), ha='center', va='bottom')
        plt.title('Average Total Passenger Extra')
        plt.xlabel('Cases')
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Total_Passenger_Extra.png', dpi = 300)
        plt.show()


# plot Average_Train_Load_Percentage 
    def plot_avg_train_load_percentage(self, avg_results: pd.DataFrame):
        avg_train_load_percentage = []
        x_ticks = ['basic scenario', 'linear passenger demand', 'hermes freight demand', 'mixed']
        alphas = [1.0, 0.7, 0.4, 0.2] 
        for row in avg_results.index:
            avg_train_load_percentage_str = avg_results.loc[row, 'Average_Train_Load_Percentage']
            avg_train_load_percentage.append(float(avg_train_load_percentage_str))

        for i in range(len(x_ticks)):
            plt.bar(x_ticks[i], avg_train_load_percentage[i], color='gray', alpha=alphas[i])

        for i, v in enumerate(avg_train_load_percentage):
            plt.text(i, v + 0.001, "{:.2f}%".format(v*100), ha='center', va='bottom')
        plt.title('Average Train Load Percentage')
        plt.xlabel('Cases')
        plt.gca().yaxis.set_major_formatter(formatter)
        y_label = plt.gca().set_ylabel('(%)', labelpad=-20)
        y_label.set_position((0, 1))
        y_label.set_rotation(0)
        plt.tight_layout()
        plt.savefig(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\Sensitivity_Analysis_Outputs\Average_Train_Load_Percentage.png', dpi = 300)
        plt.show()



############################################################################################################

test_plot = Case_Plot()
# test_plot.concat_avg_results()
test_plot.plot_all()