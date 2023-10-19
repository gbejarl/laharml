##############################
# %% 1 Import packages
# 1 Import packages
##############################

from scipy.stats import pointbiserialr
from scipy.stats import f_oneway
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from laharml import (extract_from_local_directory,
                     train_test_knn)

##############################
# %% 2 Initialize parameters
# 2 Initialize parameters
##############################

# Set training files

event_file_path = 'training.txt'
noise_file_path = 'noise.txt'


# Set data frame path and specify if reading from existing data frame

existing_data_frame = True
save_new_data_frame = False
data_frame_directory = ('/Users/gustavo/Library/CloudStorage/' +
                        'GoogleDrive-gbejarlo@mtu.edu/My Drive/Michigan Tech/' +
                        'Lahar Project/Analyses/Script performance/' +
                        'Lahar Analysis Final Results/Training Data Frames/')

# Set directory path

directory = '/Volumes/Tungurahua/FUEGO/SEISMIC'

# Set up station parameters

network = 'ZV'
station = 'FEC2'
location = ''
channel = 'HHZ'

# Set model features, parameters

features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 101]

window_length = 10  # [required] in minutes
window_overlap = 0.75  # [required] in fraction of window length

# Other parameters
decimate = 4
extended = 0
vmin = -100
vmax = 100
count_lim = 75000
freq_min = 0.1

##############################
# %% 3 Feature extraction/selection
# 3 Feature extraction/selection
##############################

data_frame_file = (network+'.'+station+'.'+location+'.'+channel+'_' +
                   str(window_length)+'_'+str(window_overlap)+'.csv')
data_frame_path = os.path.join(data_frame_directory, data_frame_file)

if not (existing_data_frame):

    # Load training dates

    s_fm = np.dtype([('station', (str, 4)), ('datestart',
                    (str, 19)), ('dateend', (str, 19))])
    s_fl = np.loadtxt(event_file_path, dtype=s_fm,
                      delimiter='\t', unpack=True, usecols=(0, 1, 2))

    if station in s_fl[0]:
        ids = np.where(s_fl[0] == station)[0]
        dtm1 = [UTCDateTime(i) for i in s_fl[1][ids]]
        dtm2 = [UTCDateTime(i) for i in s_fl[2][ids]]
    else:
        print('Station not found in training date file, try again.')

    # Define desired features

    training = pd.DataFrame()

    # Extract features

    signal_time = 0

    for i in range(len(dtm1)):
        starttime = UTCDateTime(dtm1[i])
        endtime = UTCDateTime(dtm2[i])
        dl = extract_from_local_directory(directory,
                                          network,
                                          station,
                                          location,
                                          channel,
                                          starttime,
                                          endtime,
                                          features,
                                          extended,
                                          min_freq=freq_min,
                                          decimate=decimate,
                                          window=window_length,
                                          overlap=window_overlap,
                                          plot=True,
                                          vmin=vmin,
                                          vmax=vmax,
                                          count_lim=count_lim)
        signal_time += endtime-starttime
        training = pd.concat([training, dl], ignore_index=True, sort=False)
    print('Total signal time: '+str(signal_time/3600)+' hours')

    # Include noise

    s_fm = np.dtype([('station', (str, 4)), ('datestart',
                    (str, 19)), ('type', (str, 2))])
    s_fl = np.loadtxt(noise_file_path, dtype=s_fm,
                      delimiter='\t', unpack=True, usecols=(0, 1, 2))

    if station in s_fl[0]:
        ids = np.where(s_fl[0] == station)[0]
        dtm1 = [UTCDateTime(i) for i in s_fl[1][ids]]
    else:
        print('Station not found in training date file, try again.')

    features[-1] = 100
    duration_noise_event = signal_time/len(dtm1)

    for i in range(len(dtm1)):
        starttime = UTCDateTime(dtm1[i])-(duration_noise_event/2)
        endtime = UTCDateTime(dtm1[i])+(duration_noise_event/2)
        dl = extract_from_local_directory(directory,
                                          network,
                                          station,
                                          location,
                                          channel,
                                          starttime,
                                          endtime,
                                          features,
                                          extended,
                                          min_freq=freq_min,
                                          decimate=decimate,
                                          window=window_length,
                                          overlap=window_overlap,
                                          plot=True,
                                          vmin=vmin,
                                          vmax=vmax,
                                          count_lim=count_lim)
        training = pd.concat([training, dl], ignore_index=True, sort=False)

    if save_new_data_frame:
        training.to_csv(data_frame_path)

else:

    training = pd.read_csv(data_frame_path, index_col=0)

##############################
# %% 4 Feature statistics
# 4 Feature statistics
##############################

correlation_scores = []
f_statistic = []
p_value = []

for feature in training.drop(['Times', 'Classification'], axis=1).columns:
    r, _ = pointbiserialr(training[feature], training['Classification'])
    correlation_scores.append(abs(r))

for feature in training.drop(['Times', 'Classification'], axis=1).columns:
    group1 = training.drop(['Times', 'Classification'], axis=1)[
        training['Classification'] == 0][feature]
    group2 = training.drop(['Times', 'Classification'], axis=1)[
        training['Classification'] == 1][feature]
    f_statistic.append(f_oneway(group1, group2)[0])
    p_value.append(f_oneway(group1, group2)[1])

variances = training.drop(['Times', 'Classification'], axis=1).var()

feature_statistics = pd.DataFrame({
    'Feature': training.drop(['Times', 'Classification'], axis=1).columns,
    'Correlation': correlation_scores,
    'F-statistic': f_statistic,
    'P-value': p_value,
    'Variance': variances
}).reset_index(drop=True)

feature_statistics.to_csv(data_frame_path[:-4]+'_statistics.csv')

##############################
# %% 5 Model training performance
# 5 Model training performance
##############################

# 5 min rounds

# sub_set = ["02_Envelope_5_10Hz",
#            "08_Freq_75th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "32_Skewness_Env_5_20Hz",
#            "34_Skewness_Frequency_Unfiltered",
#            "39_Entropy_Signal_Unfiltered",
#            "Times",
#            "Classification"]

# sub_set = ["00_Envelope_Unfiltered",
#            "01_Envelope_5Hz",
#            "03_Envelope_5_20Hz",
#            "04_Envelope_10Hz",
#            "05_Freq_Max_Unfiltered",
#            "06_Freq_25th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "34_Skewness_Frequency_Unfiltered",
#            "Times",
#            "Classification"]

# sub_set = ["04_Envelope_10Hz",
#            "05_Freq_Max_Unfiltered",
#            "06_Freq_25th",
#            "34_Skewness_Frequency_Unfiltered",
#            "Times",
#            "Classification"]

# 10 min rounds

# sub_set = ["01_Envelope_5Hz",
#            "08_Freq_75th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "31_Skewness_Env_5_10Hz",
#            "34_Skewness_Frequency_Unfiltered",
#            "39_Entropy_Signal_Unfiltered",
#            "Times",
#            "Classification"]

# sub_set = ["00_Envelope_Unfiltered",
#            "01_Envelope_5Hz",
#            "03_Envelope_5_20Hz",
#            "04_Envelope_10Hz",
#            "05_Freq_Max_Unfiltered",
#            "06_Freq_25th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "34_Skewness_Frequency_Unfiltered",
#            "Times",
#            "Classification"]

# sub_set = ["04_Envelope_10Hz",
#            "05_Freq_Max_Unfiltered",
#            "06_Freq_25th",
#            "34_Skewness_Frequency_Unfiltered",
#            "Times",
#            "Classification"]

# 10 min rounds with diverse noise

# sub_set = ["04_Envelope_10Hz",
#            "08_Freq_75th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "33_Skewness_Env_10Hz",
#            "34_Skewness_Frequency_Unfiltered",
#            "39_Entropy_Signal_Unfiltered",
#            "Times",
#            "Classification"]

# sub_set = ["04_Envelope_10Hz",
#            "05_Freq_Max_Unfiltered",
#            "06_Freq_25th",
#            "07_Freq_50th",
#            "08_Freq_75th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "33_Skewness_Env_10Hz",
#            "34_Skewness_Frequency_Unfiltered",
#            "39_Entropy_Signal_Unfiltered",
#            "Times",
#            "Classification"]

# sub_set = ["04_Envelope_10Hz",
#            "05_Freq_Max_Unfiltered",
#            "06_Freq_25th",
#            "34_Skewness_Frequency_Unfiltered",
#            "Times",
#            "Classification"]

# 1O min rounds with diverse noise and 50 Hz limit

# sub_set = ["03_Envelope_5_20Hz",
#            "08_Freq_75th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "33_Skewness_Env_10Hz",
#            "34_Skewness_Frequency_Unfiltered",
#            "39_Entropy_Signal_Unfiltered",
#            "Times",
#            "Classification"]

sub_set = ["04_Envelope_10Hz",
           "06_Freq_25th",
           "19_Kurtosis_Frequency_Unfiltered",
           #    "32_Skewness_Env_5_20Hz",
           "34_Skewness_Frequency_Unfiltered",
           "39_Entropy_Signal_Unfiltered",
           "Times",
           "Classification"]

sub_training = training[sub_set]

model, scaler, classification_report, confusion_matrix, neighbors = train_test_knn(sub_training,
                                                                                   scale=True,
                                                                                   get_n=True,
                                                                                   plot_n=True)

print(classification_report, confusion_matrix)
print("Neighbors: "+str(neighbors))

##############################
# %% 6 Feature plots
# 6 Feature plots
##############################

# sub_set = ["01_Envelope_5Hz",
#            "03_Envelope_5_20Hz",
#            "04_Envelope_10Hz",
#            "05_Freq_Max_Unfiltered",
#            "06_Freq_25th",
#            "07_Freq_50th",
#            "08_Freq_75th",
#            "19_Kurtosis_Frequency_Unfiltered",
#            "20_Kurtosis_Frequency_5Hz",
#            "22_Kurtosis_Frequency_5_20Hz",
#            "23_Kurtosis_Frequency_10Hz",
#            "30_Skewness_Env_5Hz",
#            "32_Skewness_Env_5_20Hz",
#            "33_Skewness_Env_10Hz",
#            "34_Skewness_Frequency_Unfiltered",
#            "36_Skewness_Frequency_5_10Hz",
#            "38_Skewness_Frequency_10Hz",
#            "39_Entropy_Signal_Unfiltered",
#            "Times",
#            "Classification"]

sub_set = ["04_Envelope_10Hz",
           "06_Freq_25th",
           "19_Kurtosis_Frequency_Unfiltered",
           #    "32_Skewness_Env_5_20Hz",
           "34_Skewness_Frequency_Unfiltered",
           "39_Entropy_Signal_Unfiltered",
           "Times",
           "Classification"]

sub_training = training[sub_set]

sns.pairplot(sub_training.drop('Times', axis=1),
             hue='Classification')

# %%
