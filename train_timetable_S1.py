import pandas as pd

timetable_df = pd.read_csv('train_timetable_S1.csv', index_col=0)
timetable_pivot = timetable_df.pivot(index='Train_ID', columns='Stop', values='Arrival_Time')