import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

total_seeds = 20
avg_intensity0p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/avg_results_intensity0.5.csv')
avg_intensity1 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/avg_results_intensity1.0.csv')
avg_intensity1p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/avg_results_intensity1.5.csv')
avg_intensity2 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/avg_results_intensity2.0.csv')
avg_intensity2p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/avg_results_intensity2.5.csv')
avg_intensity3 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/avg_results_intensity3.0.csv')
avg_intensity3p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/avg_results_intensity3.5.csv')

# avg_intensity0p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear/avg_results_intensity0.5.csv')
# avg_intensity1 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear/avg_results_intensity1.0.csv')
# avg_intensity1p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear/avg_results_intensity1.5.csv')
# avg_intensity2 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear/avg_results_intensity2.0.csv')
# avg_intensity2p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear/avg_results_intensity2.5.csv')
# avg_intensity3 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear/avg_results_intensity3.0.csv')
# avg_intensity3p5 = pd.read_csv(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_linear/avg_results_intensity3.5.csv')



# Plot distribution histogram of Delay_0_delivery (% Accepted), Delay_0_15_delivery	Delay_15_30_delivery, Delay_gt_30_delivery, Delay_0_waiting, Delay_nan_waiting(late_arrival), Delay_true_waiting

# dataframes = [avg_intensity1, avg_intensity1p5, avg_intensity2, avg_intensity2p5, avg_intensity3, avg_intensity3p5]
# for row in range(len(avg_intensity1)):
#     policy = avg_intensity1.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
#     fig, axs = plt.subplots(3, 2, figsize=(10, 10))

#     for i, df in enumerate(dataframes):
#         intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
#         delay_0_delivery = float(df.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
#         delay_0_15_delivery = float(df.loc[row,'Delay_0_15_delivery'].split(',')[1].strip())
#         delay_15_30_delivery = float(df.loc[row,'Delay_15_30_delivery'].split(',')[1].strip())
#         delay_gt_30_delivery = float(df.loc[row,'Delay_gt_30_delivery'].split(',')[1].strip())
#         delay_0_waiting = float(df.loc[row,'Delay_0_waiting'].split(',')[1].strip())
#         delay_nan_waiting = float(df.loc[row,'Delay_nan_waiting(late_arrival)'].split(',')[1].strip())
#         delay_true_waiting = float(df.loc[row,'Delay_true_waiting'].split(',')[1].strip())
#         data = [delay_0_delivery, delay_0_15_delivery, delay_15_30_delivery, delay_gt_30_delivery, delay_0_waiting, delay_nan_waiting, delay_true_waiting]
#         x_ticks = ['Delay_0_delivery', 'Delay_0_15_delivery', 'Delay_15_30_delivery', 'Delay_gt_30_delivery', 'Delay_0_waiting', 'Delay_nan_waiting', 'Delay_true_waiting']

#         # Plot data on the i-th subplot
#         ax = axs[i//2, i%2]
#         ax.bar(x_ticks, data, label=intens, alpha=0.5)
#         ax.set_title(f'{policy} - {intens}')
#         ax.set_xlabel('Delay Type')
#         ax.set_ylabel('Percentage')
#         ax.legend()
#         ax.set_xticks(range(len(x_ticks)))  # Set x-tick locations
#         ax.set_xticklabels(x_ticks, rotation=60)  # Set x-tick labels
#         ax.set_yscale('log')  # Set y-axis to logarithmic scale
#     plt.tight_layout()
#     plt.show()


############################################################################################################

df_list = [avg_intensity0p5, avg_intensity1, avg_intensity1p5, avg_intensity2, avg_intensity2p5, avg_intensity3, avg_intensity3p5]
# plot Avg_Revenue_Total_With_Intensity of all policies for each dataframes above
for row in range(len(avg_intensity1)):
    x = []
    y = []
    for df in df_list:
        policy = avg_intensity1.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
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
plt.savefig(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/Intensity_vs_Avg_Revenue_Total_With_Intensity.png', dpi = 300)
plt.show()

############################################################################################################
# Plot Imaginary_Revenue, RRT to %
for row in range(len(avg_intensity1)):
    x = []
    y = []
    for df in df_list:
        policy = avg_intensity1.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
        intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
        avg_imaginary_revenue_percentage_make = float(df.loc[row,'Imaginary_Revenue, PRT to %'].split(',')[1].strip())
        x.append(intens)
        y.append(avg_imaginary_revenue_percentage_make)

    plt.plot(x,y, label= policy) #+ '_' + 'Avg_Revenue_Total_With_Intensity'
plt.title('Intensity vs Imaginary_Revenue, PRT to %')
plt.axhline(y=1.0, color='black', linestyle='--')  # Add horizontal dashed line at y=1.0, where get 100% of imaginary revenue
plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
plt.axvline(x=2.5, color='black', linestyle='--')
plt.xlabel('Intensity')  # Label for x-axis
plt.ylabel('Imaginary_Revenue, PRT to %')  # Label for y-axis
plt.legend()
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.tight_layout()  
plt.savefig(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/Intensity_vs_Imaginary_Revenue_PRT_to_%.png', dpi = 300)
plt.show()


############################################################################################################

# Reject_All_Revenue, % to RRT
for row in range(len(avg_intensity1)):
    x = []
    y = []
    for df in df_list:
        policy = avg_intensity1.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
        intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
        avg_reject_all_revenue_percentage_make = float(df.loc[row,'Reject_All_Revenue, PRT to %'].split(',')[1].strip())
        x.append(intens)
        y.append(avg_reject_all_revenue_percentage_make)

    plt.plot(x,y, label= policy) #+ '_' + 'Avg_Revenue_Total_With_Intensity'
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
plt.savefig(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/Intensity_vs_Reject_All_Revenue_PRT%.png', dpi = 300)
plt.show()


############################################################################################################

# Plot Delay_0_delivery (% Accepted)
for row in range(len(avg_intensity1)):
    x = []
    y = []
    for df in df_list:
        policy = avg_intensity1.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
        intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
        avg_delay_0_delivery_percentage = float(df.loc[row,'Delay_0_delivery (% Accepted)'].split(',')[1].strip())
        x.append(intens)
        y.append(avg_delay_0_delivery_percentage)

    plt.plot(x,y, label= policy) #+ '_' + 'Avg_Revenue_Total_With_Intensity'
plt.title('Intensity vs Delay_0_delivery (% Accepted)')
plt.axhline(y=0.8, color='black', linestyle='--')  # Add horizontal dashed line at y=0.8, service level agreement
plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
plt.axvline(x=2.5, color='black', linestyle='--')
plt.xlabel('Intensity')  # Label for x-axis
plt.ylabel('Delay_0_delivery')  # Label for y-axis   
plt.legend()
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.tight_layout()
plt.savefig(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/Intensity_vs_Delay_0_delivery_%_Accepted.png', dpi = 300)
plt.show()


############################################################################################################


# Plot Delivery (% Total), we could integrate how many cargos into ÖPNV
for row in range(len(avg_intensity1)):
    x = []
    y = []
    for df in df_list:
        policy = avg_intensity1.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
        intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
        avg_delivery_percentage = float(df.loc[row,'Delivered (% Total)'].split(',')[1].strip()) + float(df.loc[row,'On_Train'].split(',')[1].strip())
        x.append(intens)
        y.append(avg_delivery_percentage)

    plt.plot(x,y, label= policy) #+ '_' + 'Avg_Revenue_Total_With_Intensity'
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
plt.savefig(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/Intensity_vs_Delivery_%_Total.png', dpi = 300)
plt.show()


############################################################################################################

# Plot Delay_true_waiting, % to Accepted, How many worst case
for row in range(len(avg_intensity1)):
    x = []
    y = []
    for df in df_list:
        policy = avg_intensity1.loc[row,'Seed_Time_Intensity, Policy'].split(',')[1].strip()
        intens = float(df.loc[row,'Seed_Time_Intensity, Policy'].split(',')[0].strip()[-3:])
        avg_delay_true_waiting_percentage = float(df.loc[row,'Delay_true_waiting'].split(',')[1].strip())
        x.append(intens)
        y.append(avg_delay_true_waiting_percentage)

    plt.plot(x,y, label= policy) #+ '_' + 'Avg_Revenue_Total_With_Intensity'
plt.title('Intensity vs Delay_true_waiting, % to Accepted')
plt.axvline(x=2.0, color='black', linestyle='--')  # Add vertical dashed line intensity == 2.0
plt.axvline(x=2.0, color='black', linestyle='--')
plt.xlabel('Intensity')  # Label for x-axis
plt.ylabel('Delay_true_waiting, % to Accepted')  # Label for y-axis
plt.legend()
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.tight_layout()
plt.savefig(r'D:\Nextcloud\Data\MA\Code\PyCode_MA\Outputs\STU_Time_Intensity_Selection_Output\Passenger_constant/Intensity_vs_Delay_true_waiting_%_to_Accepted.png', dpi = 300)
plt.show()
