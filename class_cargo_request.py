import numpy as np
import pandas as pd
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class STU_Request:
    stations = [0, 1, 2, 3, 4]
    # Altona, Jungfernstieg, Berliner Tor, Barmbek, Ohlsdorf 
    stops = list(range(10))
    dwell_time = 1
    train_tact = 10
    train_0_arrival_last_stop_time = 75.0
    revenue_baseline = 10
    travel_times = [0, 10, 6, 10, 6, 2, 6, 10, 6, 10]
    TW_width_set = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    TW_prob = [1/9]*9
    early_booking_set = [10, 15, 20, 25, 30, 35, 40, 45, 50]    
    np_early_booking_set = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50]) 
    # Define the mean and standard deviation, 
    # In most cases, delivery recipients TW_a start the next 30 to 45 minutes for delivery acceptance (DTW) after request released.
    mu = np.mean([25, 30, 35])
    sigma = np.std([25, 30, 35]) 
    # Generate probabilities from a normal distribution
    early_booking_prob = norm.pdf(early_booking_set, mu, sigma)
    # Normalize the probabilities so they sum to 1
    early_booking_prob /= early_booking_prob.sum()


    columns_STU = ['STU_ID', 'Arrival_Time','Assign_To_Train',
               'Load_Time','Unload_Time', 'Delay', 'Stop_o', 'Stop_d', 'PTW_a', 'PTW_b', 'PTW_width',
               'Revenue', 'Deliver_Train', 'Failed_Loading', 'Status'] 

    def __init__(self, STU_arrival_over_time: str, STU_arrival_over_station: str, random_seed: int, simulation_time: int, intensity_medium: int):
        
        self.STU_arrival_over_time  = STU_arrival_over_time         
        self.STU_arrival_over_station = STU_arrival_over_station   
        self.station_prob = self.initialize_station_prob()
        self.random_seed = random_seed
        self.simulation_time = simulation_time
        self.intensity_medium = intensity_medium
        
        # Set a random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)


    def initialize_STU_arrival_interval(self):
        if self.STU_arrival_over_time == 'constant_low':
            interval_scale = 1/(self.intensity_medium - 0.5) 
        elif self.STU_arrival_over_time == 'constant_medium':
            interval_scale = 1/(self.intensity_medium)
        elif self.STU_arrival_over_time == 'constant_high':
            interval_scale = 1/(self.intensity_medium + 0.5) 
        else:
            logging.warning('Invalid STU arrival mode.')
        logging.warning(f'Interval Scale: {interval_scale}')
        return  interval_scale

 
    def initialize_station_prob(self):
        if self.STU_arrival_over_station == 'uniform':
            # Constant station probability as representative case
            # [Altona, Jungfernstieg, Berliner Tor, Barmbek, Ohlsdorf] 
            station_prob = [0.2, 0.2, 0.2, 0.2, 0.2]
        elif self.STU_arrival_over_station == 'hermes_peaks':
            # Cargo data from Hermes in Hamburg inner city: 
            # One Cargo Peak Sation: Barmbek, small Cargo Peak Altona
            # https://newsroom.hermesworld.com/wp-content/uploads/heatmaps2019/index.html#hamburg
            cargo_df = pd.read_csv('cargo_data.csv', index_col=0)
            total_cargo = cargo_df['Cargos'].sum()
            cargo_df['Cargo_Percentage'] = cargo_df['Cargos']/total_cargo
            station_prob = cargo_df['Cargo_Percentage'].to_list()
            # [0.22348367029548988, 0.14914463452566096, 0.12779937791601867, 0.31912908242612753, 0.18044323483670296]
            # [Altona, Jungfernstieg, Berliner Tor, Barmbek, Ohlsdorf] 
        else:
            raise ValueError('Invalid STU arrival mode.')
        return station_prob
    

    def generate_STU_requests_df(self):
        interval_scale = self.initialize_STU_arrival_interval()
        arrival_times = self.generate_arrival_times(interval_scale)
        STU_total = len(arrival_times)
        logging.warning(f'Total STU requests: {STU_total}')
        request_ids = [x for x in range(STU_total)]

        station_pairs = self.generate_request_pairs(STU_total)
        stop_os, stop_ds = self.generate_stops(station_pairs)
        TW_as, TW_bs, TW_widths = self.generate_dtws(arrival_times)
        PTW_as, PTW_bs, PTW_ws = self.generate_ptws(TW_as, TW_bs, TW_widths, stop_os, stop_ds)
        paths = self.generate_paths(station_pairs)
        # print(paths)
        revenues = self.generate_revenues(arrival_times, PTW_as, PTW_ws)

        STU_requests_df = pd.DataFrame(columns=STU_Request.columns_STU)
        for i, a_t, so, sd, p1, p2, width, r in zip(request_ids, arrival_times, stop_os, stop_ds, PTW_as, PTW_bs, PTW_ws, revenues):
            STU_requests_df.loc[i, ['STU_ID', 'Arrival_Time','Stop_o', 'Stop_d', 'PTW_a', 'PTW_b', 'PTW_width','Revenue']] = [i, a_t, so, sd, p1, p2, width, r]

        return STU_requests_df
    
    def plot_revenues_distribution(self, STU_requests_df):
        sns.histplot(STU_requests_df['Revenue'], bins=30, kde=True)
        plt.title('Revenue Distribution')
        plt.xlabel('Revenue')
        plt.ylabel('Frequency')
        plt.show()


    def generate_arrival_times(self, interval_scale):
        arrival_times = []
        a_t = STU_Request.train_0_arrival_last_stop_time
        while a_t < self.simulation_time:
            a_interval = np.random.exponential(scale = interval_scale)
            a_t += a_interval
            if a_t >= self.simulation_time - 1:
                break
            else:
                arrival_times.append(np.round(a_t, 0))
        return arrival_times

    # Function to generate request pairs
    def generate_request_pairs(self, STU_total):
        request_pairs = []
        for _ in range(STU_total):
            # Randomly select two distinct elements with given probabilities
            pair = np.random.choice(STU_Request.stations, 2, replace=False, p=self.station_prob)
            request_pairs.append(tuple(pair))
        return request_pairs


    def generate_dtws(self, arrival_times):
        TWas = []
        TWbs = []
        TW_widths = []
        for a_t in arrival_times:
            TW_w = np.random.choice(STU_Request.TW_width_set, p= STU_Request.TW_prob)
            TW_a = a_t + np.random.choice(STU_Request.early_booking_set, p= STU_Request.early_booking_prob)
            TW_b = TW_a + TW_w
            TWas.append(TW_a)
            TWbs.append(TW_b)
            TW_widths.append(TW_w)
        return TWas, TWbs, TW_widths   

    def generate_paths(self, request_pairs):
        paths = []
        for pair in request_pairs:        
            origin, destination = pair
            if origin < destination:
                paths.append([x for x in range(origin, destination+1)])
            else:
                paths.append([x for x in range(2*max(STU_Request.stations)-(origin-1), 2*max(STU_Request.stations)-(destination-1)+1)])
        return paths 

    def generate_stops(self, request_pairs):
        stop_os = []
        stop_ds = []
        for pair in request_pairs:        
            origin, destination = pair
            if origin < destination:
                stop_os.append(origin)
                stop_ds.append(destination)
            else:
                stop_os.append(2*max(STU_Request.stations)-(origin-1))
                stop_ds.append(2*max(STU_Request.stations)-(destination-1))            
        return stop_os, stop_ds    
    

    def generate_ptws(self, TWas, TWbs, TW_widths, stop_os, stop_ds):
        PTWas = []
        PTWbs = []
        PTW_widths = []
        for twa, twb, tww, so, sd in zip(TWas, TWbs, TW_widths, stop_os, stop_ds):
            #print(f'TW_a, TW_b, TW_width, stop_o, stop_d: {twa, twb, tww, so, sd}')
            PTW_w = tww
            travel_o_d = sum(STU_Request.travel_times[so+1:sd+1]) + STU_Request.dwell_time*(sd-so+1)
            PTW_a = twa - travel_o_d
            PTW_b = PTW_a + PTW_w
            PTWas.append(PTW_a)
            PTWbs.append(PTW_b)
            PTW_widths.append(PTW_w)
            #print(f'PTW_a, PTW_b, PTW_width: {PTW_a, PTW_b, PTW_w}')
        return PTWas, PTWbs, PTW_widths
    

    def generate_revenues(self, arrival_times, PTW_as, PTW_ws):
        revenues = []
        for a_t, ptw_a, ptw_w in zip(arrival_times, PTW_as, PTW_ws):
            # revenue per STU = fare per STU (dynamic) - operating cost per STU(constant)
            # revenue per STU is proportional to fare per STU ==> revenue is dynamic
            # revenue depends on how cargo request arrival time and the size of TW
            # ptw_a depends on TW_a and travel time
            # ptw_a is small means either TW_a is small or travel time is large (cargo occupy capacity for a longer time) or both
            # residual is negative means, the cargo request arrives after its PTW_a, effective TW is smaller than TW_width, get late arrival penalty
            residual = ptw_a - a_t
            if residual < 0: # ptw_a already execeeds a_t, less chance to reorganize
                r = STU_Request.revenue_baseline + abs(residual/STU_Request.train_tact)*2 - (ptw_w/STU_Request.train_tact)*1
            else: # full use of processing time window
                r = STU_Request.revenue_baseline - (ptw_w/STU_Request.train_tact)*1
            revenues.append(r)
        return revenues

# Debugging Test:
# stu_request_instance = STU_Request(STU_arrival_over_time = 'constant_medium', STU_arrival_over_station = 'hermes_peaks', random_seed = 2024, simulation_time = 255, intensity_medium = 2.25)
# STU_requests_df = stu_request_instance.generate_STU_requests_df()
# revenues_distribution = stu_request_instance.plot_revenues_distribution(STU_requests_df)
# # STU_requests_df[STU_requests_df['Revenue'] > 9]
# np.mean(STU_requests_df['Revenue'])
# # min(STU_requests_df['Revenue'])


############################################################################################################################################################################
## plot the station probability distribution

# hermes_peaks_prob = [0.22348367029548988, 0.14914463452566096, 0.12779937791601867, 0.31912908242612753, 0.18044323483670296]
# uniform_prob = [0.2, 0.2, 0.2, 0.2, 0.2]
# S1_stations = ['Altona', 'Jungfernstieg', 'Berliner Tor', 'Barmbek', 'Ohlsdorf'] 

# bar_width = 0.35  # Set the width of the bars
# index = np.arange(len(S1_stations))  # the x locations for the groups

# # Create bars for uniform_prob
# bars1 = plt.bar(index, uniform_prob, bar_width, color='gray', alpha=0.3, label='Uniform Probability')

# # Create bars for hermes_peaks_prob
# bars2 = plt.bar(index + bar_width, hermes_peaks_prob, bar_width, color='gray', alpha=0.7, label='Hermes Peaks Probability')

# # plt.xlabel('S1 Stations')
# # plt.ylabel('Probability')
# plt.title('Station Probability Distribution', fontsize=8)
# plt.xticks(index + bar_width / 2, S1_stations, fontsize=8)  # Center x-axis labels for better visibility
# plt.yticks([0.0, 0.10, 0.20, 0.30], fontsize=8)  # Set y-ticks
# # Set legend names and position
# plt.legend((bars1[0], bars2[0]), ('Uniform','Hermes Peaks'), loc='upper left', fontsize=8)

# # Add uniform_prob values on top of each bar
# for bar, prob in zip(bars1, uniform_prob):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+0.005, f'{round(prob * 100, 2)}%', ha='center', va='bottom', fontsize=8)

# # Add hermes_peaks_prob values on top of each bar
# for bar, prob in zip(bars2, hermes_peaks_prob):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+0.005, f'{round(prob * 100, 2)}%', ha='center', va='bottom', fontsize=8)

# plt.ylim(0, 0.36)
# plt.tight_layout()
# plt.savefig('D:\\Nextcloud\\Data\\MA\\Code\\PyCode_MA\\Outputs\\station_departure_destination_probability_distribution.png', dpi=300)
# plt.show()