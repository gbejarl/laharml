##############################
# %% 1 Import packages
# 1 Import packages
##############################

import os
import obspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from obspy import UTCDateTime
from itertools import chain
from scipy import signal

##############################
# %% 2 Initialize parameters
# 2 Initialize parameters
##############################

# Set station information

reference = 'Las Lajas'
network = 'GI'
station = 'FG12'
location = '00'
channel = 'BHZ'
calibration = [2.9947e8, 4.81e8]

# Set detection results directory

detections = '/Users/gustavo/Library/CloudStorage/GoogleDrive-gbejarlo@mtu.edu/My Drive/Michigan Tech/Lahar Project/Analyses/Script performance/Lahar Analysis Final Results/Raw Detections/'

# Set manual record filepath

observations = '/Users/gustavo/Library/CloudStorage/GoogleDrive-gbejarlo@mtu.edu/My Drive/Michigan Tech/Lahar Project/Analyses/Script performance/Lahar Analysis Final Results/record_laslajas_manual.csv'

##############################
# %% 3 Script
# 3 Script
##############################

column_names = ['event_id', station+'_start', station+'_end']

observations = pd.read_csv(observations,
                           index_col=0,
                           usecols=column_names).dropna()

start_time1 = pd.to_datetime(observations[column_names[1]].iloc[0])
end_time1 = pd.to_datetime(observations[column_names[2]].iloc[-1])

all_runs = []

for filename in sorted(os.listdir(os.path.join(detections, station))):
    if filename.endswith('.txt'):
        all_runs.append(pd.read_csv(os.path.join(detections, station, filename),
                                    index_col=None, header=None))

for a in all_runs:
    a[0] = pd.to_datetime(a[0])
    a[1] = pd.to_datetime(a[1])

rates = []

for a in all_runs:
    start_time2 = pd.to_datetime(a[0].iloc[0])
    end_time2 = pd.to_datetime(a[1].iloc[-1])

    start_time = min(start_time1, start_time2)
    end_time = max(end_time1, end_time2)

    intervals = pd.date_range(start_time, end_time, freq='10min')
    category = np.zeros(len(intervals))

    cat_blank = pd.DataFrame({'datetime': intervals, 'zeros': category})

    cat_obs = cat_blank.copy()
    for b, c in observations.iterrows():
        cat_obs.loc[(cat_obs['datetime'] >= c[0]) & (
            cat_obs['datetime'] <= c[1]), 'zeros'] = 1

    cat_det = cat_blank.copy()
    for b, c in a.iterrows():
        cat_det.loc[(cat_det['datetime'] >= c[0]) & (
            cat_det['datetime'] <= c[1]), 'zeros'] = 1

    observed = [bool(num) for num in cat_obs['zeros'].to_list()]
    detected = [bool(num) for num in cat_det['zeros'].to_list()]

    TP = TN = FP = FN = 0

    for obs, det in zip(observed, detected):
        if obs and det:
            TP += 1
        elif not obs and not det:
            TN += 1
        elif not obs and det:
            FP += 1
        elif obs and not det:
            FN += 1

    # Calculate True Positive Rate (TPR)
    TPR = TP / (TP + FN)

    # Calculate False Positive Rate (FPR)
    FPR = FP / (FP + TN)

    rates.append([TPR, FPR])

    print(f"True Positive Rate: {TPR}")
    print(f"False Positive Rate: {FPR}\n")

# Extract TPRs and FPRs from the rates list
TPRs, FPRs = zip(*rates)

# Create the ROC curve
plt.plot(FPRs, TPRs, label='ROC curve')

# Add labels and a legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])

# Display the plot
plt.show()
# %%
