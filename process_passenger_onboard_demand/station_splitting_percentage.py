import pandas as pd
import numpy as np
import process_passenger_onboard_demand.S1_baseline_passenger_demand as S1_B_P

S1_P = S1_B_P.S1_P
S1_W = S1_B_P.S1_W

S1 = pd.read_csv(rf'D:\Nextcloud\Data\MA\Code\PyCode_MA\S1_Passenger_Data_2019.csv', encoding='latin-1')
# drop rows from Stops 'Rissen' to ''Bahrenfeld'
S1 = S1.drop(S1.index[0:8])
# reverse rows last row becons first row and first row becons last row, reset index
S1 = S1.iloc[::-1].reset_index(drop=True)

# Passenger Demand Spliting to stops A and B
# calculate percentage of Embarking_P and Embarking_W based on the total number of Embarking_P + Embarking_W
embarking_p = S1['Embarking_P'] / (S1['Embarking_P'] + S1['Embarking_W'])
embarking_w = S1['Embarking_W'] / (S1['Embarking_P'] + S1['Embarking_W'])
# Direction Poppenbüttel is direction Ohlsdorf, Direction Wedel is direction Altona
spliting_df = pd.DataFrame([embarking_p, embarking_w], index=['Embarking_O', 'Embarking_A']).T
station_splitting_percentage_O = pd.DataFrame()
station_splitting_percentage_A = pd.DataFrame()
# Get selected stops data for stops 'Ohlsdorf', 'Barmbek', 'Berliner Tor', 'Junfernstieg', 'Altona'
station_splitting_percentage_O['Embarking'] = spliting_df.loc[[5, 8, 13, 15, 20], 'Embarking_O'].reset_index(drop=True)
station_splitting_percentage_O = station_splitting_percentage_O.iloc[::-1].reset_index(drop=True)

station_splitting_percentage_A['Embarking']= spliting_df.loc[[5, 8, 13, 15, 20], 'Embarking_A'].reset_index(drop=True)

#Get spliting percentage with stop information
station_splitting_percentage_O = pd.concat([S1_P['Stops'], station_splitting_percentage_O], axis=1)
station_splitting_percentage_A = pd.concat([S1_W['Stops'], station_splitting_percentage_A], axis=1)


station_splitting_percentage = pd.DataFrame()
station_splitting_percentage['Stops'] = pd.concat([station_splitting_percentage_O['Stops'], station_splitting_percentage_A['Stops']], axis=0).reset_index(drop=True)
station_splitting_percentage['Embarking_Percentage'] = pd.concat([station_splitting_percentage_O['Embarking'], station_splitting_percentage_A['Embarking']], axis=0).reset_index(drop=True)


#############################################################################################################