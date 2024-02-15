import simpy
import pandas as pd

# Simulate a single train's journey
def simulate_train_journey(env, train_id, stops, travel_times, dwell_time, arrival_events):
    for stop, travel_time in zip(stops, travel_times):
        yield env.timeout(travel_time)
        arrival_time = env.now
        arrival_events.append((train_id, stop, arrival_time))
        yield env.timeout(dwell_time)

# Generate a timetable for multiple trains
def generate_train_timetable(env, num_trains, stops, travel_times, dwell_time, arrival_rate, arrival_events):
    for train_id in range(num_trains):
        env.process(simulate_train_journey(env, train_id, stops, travel_times, dwell_time, arrival_events))
        yield env.timeout(arrival_rate)

# Set up the simulation environment and run the simulation
def run_simulation(num_trains, stops, travel_times, dwell_time, arrival_rate, simulation_time):
    env = simpy.Environment()
    arrival_events = []
    env.process(generate_train_timetable(env, num_trains, stops, travel_times, dwell_time, arrival_rate, arrival_events))
    env.run(until=simulation_time)
    return arrival_events

# Convert the list of arrival events into a DataFrame
def create_timetable_dataframe(arrival_events):
    df = pd.DataFrame(arrival_events, columns=['Train_ID', 'Stop', 'Arrival_Time'])
    return df

# Network details
# Stops: Altona_A, Jungfernstieg_A, Berliner Tor_A, Barmbek_A, Ohlsdorf_A, Ohlsdorf_B, Barmbek_B, Berliner Tor_B, Jungfernstieg_B, Altona_B
stops = list(range(10))
travel_times = [0, 10, 6, 10, 6, 2, 6, 10, 6, 10]
dwell_time = 1
arrival_rate = 10  # Trains every 10 minutes
num_trains = 100  
simulation_time = 300  # Simulation time in minutes

# Run the simulation
arrival_events = run_simulation(num_trains, stops, travel_times, dwell_time, arrival_rate, simulation_time)

# Create timetable DataFrame
timetable_df = create_timetable_dataframe(arrival_events)
timetable_df.to_csv('train_timetable_S1.csv')

# Pivot the timetable DataFrame to show arrival times for each stop by train
timetable_pivot = timetable_df.pivot(index='Train_ID', columns='Stop', values='Arrival_Time')
timetable_pivot.to_csv('train_timetable_pivot_S1.csv')