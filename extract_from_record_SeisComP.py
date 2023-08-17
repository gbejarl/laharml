##############################
#%% 1 Import packages
# 1 Import packages
##############################

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time


from laharml import (SeisComP_download_extract,
                     train_test_knn, 
                     predict_knn, 
                     clean_detections, 
                     retrieve_dates, 
                     plot_detections)

##############################
#%% 2 Initialize parameters
# 2 Initialize parameters
##############################

# Set date range

start_date = '2022-05-15'
end_date = '2022-06-01'

# Set up client/server

# reserved line
# reserved line
server_address = "http://172.20.8.4:8080"

# Set up station parameters

network = 'GI'
station = 'FG16'
location = '00'
channel = '*Z'

# Set model features, parameters

features = [3,4,5,11]
window_length = 10 # [required] in minutes
window_overlap = 0.5 # [required] in fraction of window length

# Set output

output_folder_path = ""
show_detections = True
save_log = True
save_detections = True

# Other optional parameters
decimation_factor = [] # decimate data by this factor
minimum_frequency = [] # in Hz
maximum_frequency = [] # in Hz

##############################
#%% 3 Script
# 3 Script
##############################

dt1 = UTCDateTime(start_date)
dt2 = UTCDateTime(end_date)

link = server_address+"/fdsnws/dataselect/1/query?"

# reserved line
# reserved line
# reserved line
# reserved line

# Load training dates

s_fm = np.dtype([('station',(str,4)),('datestart',(str,19)),('dateend',(str,19))])
s_fl = np.loadtxt('training.txt',dtype=s_fm,delimiter='\t',unpack=True,usecols=(0,1,2))

if station in s_fl[0]:
    ids = np.where(s_fl[0]==station)[0]
    dtm1 = [UTCDateTime(i) for i in s_fl[1][ids]]
    dtm2 = [UTCDateTime(i) for i in s_fl[2][ids]]
else:
    print('Station not found in training date file, try again.')

# Model training

features_class = features.copy()
features_class.append(12)
training = pd.DataFrame()

for i in range(len(dtm1)):
    starttime = UTCDateTime(dtm1[i])
    endtime = UTCDateTime(dtm2[i])
    extended = (endtime-starttime)/3600/2
    dl = SeisComP_download_extract(link,network,station,location,channel,starttime,endtime,features_class,extended,window=window_length,overlap=window_overlap,decimate=decimation_factor,min_freq=minimum_frequency,max_freq=maximum_frequency,plot=True)
    training = pd.concat([training,dl], ignore_index=True, sort=False)

model,scaler,classification_report,confusion_matrix,neighbors = train_test_knn(training,
                                                                               scale=True,
                                                                               get_n=True,
                                                                               plot_n=True)

print('''Training model with {var_n} neighbors:
Correct positive predictions are {var_1} out of {var_2} total positive predictions.
Correct negative predictions are {var_3} out of {var_4} total negative predictions.
'''.format(var_n=neighbors,
           var_1=confusion_matrix[0,0],
           var_2=confusion_matrix[0,0]+confusion_matrix[0,1],
           var_3=confusion_matrix[1,1],
           var_4=confusion_matrix[1,1]+confusion_matrix[1,0]))

# Datetime rolling

x1 = np.array([])
x2 = np.array([])

starttime = dt1

while starttime < dt2:

    endtime = starttime + (3600*24)

    print('\rStarting '+starttime.strftime('%Y-%m-%dT%H:%M:%S')+'----\n',end="", flush=True)

    try:
        unclassified_data_frame,st = SeisComP_download_extract(link,network,station,location,channel,starttime,endtime,features,window=window_length,overlap=window_overlap,keep=True)
        classified_data_frame = predict_knn(unclassified_data_frame,model,scaler=scaler)
        cleaned_data_frame = clean_detections(classified_data_frame)
        lah_0,lah_1,lah_0l,lah_1l = retrieve_dates(cleaned_data_frame)
        lah_count = len(lah_0)
        x1 = np.append(x1,lah_0)
        x2 = np.append(x2,lah_1)

        if (show_detections == True) and (len(lah_0) > 0):
                    plot_detections(cleaned_data_frame,st,show=True,save=False,target='Detection',vmin=-125,vmax=125,count_lim=200000)

        if save_log:
            xts = np.stack(([i.strftime('%Y-%m-%dT%H:%M:%S') for i in x1],
                        [i.strftime('%Y-%m-%dT%H:%M:%S') for i in x2]),axis=-1)
            out_log = 'log_'+station+'_'+dt1.strftime('%Y%m%d')+'_'+dt2.strftime('%Y%m%d')+'.txt'    
            np.savetxt(out_log, xts, delimiter=",",fmt='%s')
    except:
        xts = []
        lah_count = 0
        print('\rNo data, moving to next dates---',end="", flush=True)
        time.sleep(2)

    if np.any(x1):
        tot_count = len(x1)
    else:
         tot_count = 0

    starttime = starttime+(3600*12)

    print('''\r\n
          Finished detections for the following dates:
          {date_1} to {date_2}.
          Detections found = {number_1}
          Number of detections saved = {number_2}  
          '''.format(date_1=starttime.strftime('%Y-%m-%dT%H:%M:%S'),
                     date_2=endtime.strftime('%Y-%m-%dT%H:%M:%S'),
                     number_1=lah_count,
                     number_2=tot_count))
    
# Final dates

passed_x1 = []
passed_x2 = []

xts = np.loadtxt(out_log,delimiter=',',dtype=str,unpack=True)

if not(np.size(xts)==0):

    # Remove detection of less than 30 minutes, duplicates

    for i in range(len(xts)):
        if (UTCDateTime(xts[i][1])-UTCDateTime(xts[i][0])) > (3600*0.5):
            passed_x1.append(xts[i][0])
            passed_x2.append(xts[i][1])

    passed_x1 = np.array(passed_x1)
    passed_x2 = np.array(passed_x2)

    # Remove overlapping detections keeping the longest

    for i in range(len(passed_x1)):
        skipped = [*range(len(passed_x1))]
        skipped.remove(i)
        for j in skipped:
            if (UTCDateTime(passed_x1[i])<=UTCDateTime(passed_x2[j])) and (UTCDateTime(passed_x2[i])>=UTCDateTime(passed_x1[j])):
                if (UTCDateTime(passed_x2[i])-UTCDateTime(passed_x1[i]))<(UTCDateTime(passed_x2[j])-UTCDateTime(passed_x1[j])):
                    passed_x1[i] = passed_x1[j]
                    passed_x2[i] = passed_x2[j]

    x1 = np.unique(np.array(passed_x1))
    x2 = np.unique(np.array(passed_x2))

else:

    x1 = np.array([])
    x2 = np.array([])

# Save final dates

if save_detections and (not(np.size(x1))==0):
    
        xts = np.stack((x1,x2),axis=-1)
        out_dts = 'detection_'+station+'_'+UTCDateTime(x1[0]).strftime('%Y%m%d')+'_'+UTCDateTime(x2[-1]).strftime('%Y%m%d')+'.txt'    
        np.savetxt(out_dts, xts, delimiter=",",fmt='%s')
# %%
