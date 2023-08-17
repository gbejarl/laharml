import os
import obspy
import sklearn
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy import signal
from obspy import UTCDateTime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% fdsn_download_extract

def FDSN_download_extract(client,
                          network,
                          station,
                          location,
                          channel,
                          starttime,
                          endtime,
                          features,
                          extended=0,
                          window=5,
                          overlap=0.25, 
                          decimate=[],
                          min_freq=[],
                          max_freq=[],
                          keep=False,
                          plot=False,
                          vmin=None,
                          vmax=None):
    
    feature_labels = ['Envelope_Unfiltered',
                      'Envelope_5Hz',
                      'Envelope_5_10Hz',
                      'Envelope_10Hz',
                      'Freq_Max_Unfiltered',
                      'Freq_25th',
                      'Freq_50th',
                      'Freq_75th',
                      'Kurtosis_Time',
                      'Kurtosis_Freq',
                      'Skewness_Time',
                      'Skewness_Freq',
                      'Classification']
    
    print('\rDownloading data stream---------',end="", flush=True)
    
    st = client.get_waveforms(network,station,location,channel,
                                starttime=starttime-(extended*3600),
                                endtime=endtime+(extended*3600))
    
    stored_stream = st.copy()
    
    print('\rPreprocessing-------------------',end="", flush=True)
    
    st = st.merge(fill_value='interpolate').detrend('linear')
    st = st.select(component='Z')[0]

    if decimate:
        st.decimate(factor=decimate)

    if min_freq or max_freq:
        if min_freq and max_freq:
            st.filter('bandpass',freqmin=min_freq,freqmax=max_freq)
        elif min_freq:
                st.filter('highpass',freq=min_freq)
        elif max_freq:
                st.filter('lowpass',freq=max_freq)
    
    st_x = st.data
    st_t = st.times(type='timestamp')
    times = np.array([])

    if plot:
        st_m = st.times(type='matplotlib')
        isamp = int(1/(st.stats.delta))
        inseg = isamp*10
        infft = 2048

        fig, ax = plt.subplots(2,1,sharex=True,figsize=(15,7))
        ax[0].plot(st_m,st_x,lw=0.5,c='k')
        ax[0].set_xlim(min(st_m),max(st_m))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax[0].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
        ax[0].set_title(station)
        xspf,xspt,xsps=signal.spectrogram(st_x,fs=isamp,nperseg=inseg,nfft=infft,detrend=False)
        xspd = 20*np.log10(abs(xsps))
        ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        im=ax[1].imshow(xspd,origin='lower',interpolation='nearest',aspect='auto',
                    extent=[st_m[0],st_m[-1],xspf[0],xspf[-1]],
                    cmap='jet',vmin=vmin,vmax=vmax)
        ax[1].set_ylabel('Hz')
        plt.tight_layout
        plt.show()

    # Conditionals 1

    print('\rGenerating empty arrays---------',end="", flush=True)

    if 0 in features:
        en0 = abs(obspy.signal.filter.envelope(st_x))
        envelope0 = np.array([])

    if 1 in features:
        st1 = st.copy()
        st1.filter('highpass',freq=5)
        am1 = st1.data
        en1 = abs(obspy.signal.filter.envelope(am1))
        envelope1 = np.array([])

    if 2 in features:
        st2 = st.copy()
        st2.filter('bandpass',freqmin=5,freqmax=10)
        am2 = st2.data
        en2 = abs(obspy.signal.filter.envelope(am2))
        envelope2 = np.array([])

    if 3 in features:
        st3 = st.copy()
        st3.filter('highpass',freq=10)
        am3 = st3.data
        en3 = abs(obspy.signal.filter.envelope(am3))
        envelope3 = np.array([])

    if 4 in features:
        freqmax_unfiltered = np.array([])

    if 5 in features:
        freq25th = np.array([])

    if 6 in features:
        freq50th = np.array([])

    if 7 in features:
        freq75th = np.array([])

    if 8 in features:
        kurtosis_time = np.array([])

    if 9 in features:
        kurtosis_freq = np.array([])

    if 10 in features:
        skewness_time = np.array([])

    if 11 in features:
        skewness_freq = np.array([])

    segleng=20*(1/st.stats.delta)
    nperseg=2**np.ceil(np.log2(segleng))
    nfft=4*nperseg
    ww = (1/(st.stats.delta))*60*window
    ll = ww*(1-overlap)
    
    print('\rPopulating arrays---------------',end="", flush=True)

    for a in np.arange(0,(len(st_x))-ww,ll):
                
        times = np.append(times,st_t[int(a+(ww/2))]) # Times
        sub = st_x[int(a):int(a+ww)] # Subset unfiltered

        if 0 in features:
            envelope0 = np.append(envelope0,np.mean(en0[int(a):int(a+ww)]))

        if 1 in features:
            envelope1 = np.append(envelope1,np.mean(en1[int(a):int(a+ww)]))

        if 2 in features:
            envelope2 = np.append(envelope2,np.mean(en2[int(a):int(a+ww)]))

        if 3 in features:
            envelope3 = np.append(envelope3,np.mean(en3[int(a):int(a+ww)]))

        if (4 in features) or (5 in features) or (6 in features) or (7 in features):
            ff,pp = signal.welch(sub,fs=(1/st.stats.delta),window='hann',
                    nperseg=nperseg,noverlap=nperseg/2,nfft=nfft)
            if 4 in features:
                freqmax_unfiltered = np.append(freqmax_unfiltered,ff[np.argmax(pp)])
            csd=np.cumsum(pp)
            csd=csd-np.min(csd[1:])
            csd=csd/csd.max()
            if 5 in features:
                idx=np.argmin(np.abs(csd-.25))
                freq25th = np.append(freq25th,ff[idx])
            if 6 in features:
                idx=np.argmin(np.abs(csd-.50))
                freq50th = np.append(freq50th,ff[idx])
            if 7 in features:
                idx=np.argmin(np.abs(csd-.75))
                freq75th = np.append(freq75th,ff[idx])
        
        if 8 in features:
            kurtosis_time = np.append(kurtosis_time,stats.kurtosis(sub))

        if 9 in features:
            kurtosis_freq = np.append(kurtosis_freq,stats.kurtosis(pp))

        if 10 in features:
            skewness_time = np.append(skewness_time,stats.skew(sub))

        if 11 in features:
            skewness_freq = np.append(skewness_freq,stats.skew(pp))
        
        print(f'\r{100*((a+1)/(len(st_x)-ww+1)):3.2f}% done--------------------',
              end='',
              flush=True)

    print('\r100.00 % done-------------------\nFeatures extracted', end='', flush=True)
    
    data_frame = pd.DataFrame({'Times':times}) 

    if 12 in features:    
        cat = np.zeros(times.shape)
        cat[(times>=starttime) & (times<=endtime)] = 1
        data_frame[feature_labels[12]] = cat
      
    if 11 in features:
        data_frame.insert(loc=0,column=feature_labels[11],value=skewness_freq)
    if 10 in features:
        data_frame.insert(loc=0,column=feature_labels[10],value=skewness_time)
    if 9 in features:
        data_frame.insert(loc=0,column=feature_labels[9],value=kurtosis_freq)
    if 8 in features:
        data_frame.insert(loc=0,column=feature_labels[8],value=kurtosis_time)
    if 7 in features:
        data_frame.insert(loc=0,column=feature_labels[7],value=freq75th)
    if 6 in features:
        data_frame.insert(loc=0,column=feature_labels[6],value=freq50th)
    if 5 in features:
        data_frame.insert(loc=0,column=feature_labels[5],value=freq25th)
    if 4 in features:
        data_frame.insert(loc=0,column=feature_labels[4],value=freqmax_unfiltered)
    if 3 in features:
        data_frame.insert(loc=0,column=feature_labels[3],value=envelope3)
    if 2 in features:
        data_frame.insert(loc=0,column=feature_labels[2],value=envelope2)
    if 1 in features:
        data_frame.insert(loc=0,column=feature_labels[1],value=envelope1)
    if 0 in features:
        data_frame.insert(loc=0,column=feature_labels[0],value=envelope0)

    if keep:
        return data_frame, stored_stream
    else:
        return data_frame

#%%

def SeisComP_download_extract(head='',
                              network='',
                              station='',
                              location='',
                              channel='',
                              starttime=[],
                              endtime=[],
                              features=[],
                              extended=0,
                              window=5,
                              overlap=0.25,
                              decimate=[],
                              min_freq=[],
                              max_freq=[],
                              keep=False,
                              plot=False,
                              vmin=None,
                              vmax=None):
    
    feature_labels = ['Envelope_Unfiltered',
                    'Envelope_5Hz',
                    'Envelope_5_10Hz',
                    'Envelope_10Hz',
                    'Freq_Max_Unfiltered',
                    'Freq_25th',
                    'Freq_50th',
                    'Freq_75th',
                    'Kurtosis_Time',
                    'Kurtosis_Freq',
                    'Skewness_Time',
                    'Skewness_Freq',
                    'Classification']
    
    if starttime and endtime:

        t1 = starttime - (extended*3600)
        t2 = endtime + (extended*3600)

        str_starttime = (f'{t1.year:04d}-'+
                        f'{t1.month:02d}-'+
                        f'{t1.day:02d}T'+
                        f'{t1.hour:02d}%3A'+
                        f'{t1.minute:02d}%3A'+
                        f'{t1.second:02d}')
        str_endtime = (f'{t2.year:04d}-'+
                    f'{t2.month:02d}-'+
                    f'{t2.day:02d}T'+
                    f'{t2.hour:02d}%3A'+
                    f'{t2.minute:02d}%3A'+
                    f'{t2.second:02d}')
    
    url1 = head
    url2 = 'starttime='+str_starttime+'&'
    url3 = 'endtime='+str_endtime+'&'
    url4 = 'network='+network+'&'
    url5 = 'station='+station+'&'
    url6 = 'location='+location+'&'
    url7 = 'channel='+channel+'&'
    url8 = 'nodata='+'404'

    url = url1

    if starttime:
        url+=url2
    if endtime:
        url+=url3
    if network != '':
        url+=url4
    if station != '':
        url+=url5
    if location != '':
        url+=url6
    if channel != '':
        url+=url7
    
    url+=url8

    #path = '/temp'
    #Path(path).mkdir(parents=True, exist_ok=True)

    fn = 'temp_stream.mseed'
    ff = requests.get(url)
    with open(fn, 'wb') as f:
        f.write(ff.content)

    st = obspy.read(fn)

    stored_stream = st.copy()

    st = st.merge(fill_value='interpolate').detrend('linear')
    st = st.select(component='Z')[0]
    
    if decimate:
        st.decimate(factor=decimate)

    if min_freq or max_freq:
        if min_freq and max_freq:
            st.filter('bandpass',freqmin=min_freq,freqmax=max_freq)
    elif min_freq:
        st.filter('highpass',freq=min_freq)
    elif max_freq:
        st.filter('lowpass',freq=max_freq)

    st_x = st.data
    st_t = st.times(type='timestamp')
    times = np.array([])

    if plot:
        st_m = st.times(type='matplotlib')
        isamp = int(1/(st.stats.delta))
        inseg = isamp*10
        infft = 2048

        fig, ax = plt.subplots(2,1,sharex=True,figsize=(15,7))
        ax[0].plot(st_m,st_x,lw=0.5,c='k')
        ax[0].set_xlim(min(st_m),max(st_m))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax[0].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
        ax[0].set_title(station)
        xspf,xspt,xsps=signal.spectrogram(st_x,fs=isamp,nperseg=inseg,nfft=infft,detrend=False)
        xspd = 20*np.log10(abs(xsps))
        ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        im=ax[1].imshow(xspd,origin='lower',interpolation='nearest',aspect='auto',
                    extent=[st_m[0],st_m[-1],xspf[0],xspf[-1]],
                    cmap='jet',vmin=vmin,vmax=vmax)
        ax[1].set_ylabel('Hz')
        plt.tight_layout
        plt.show()

    # Conditionals 1

    print('\rGenerating empty arrays---------',end="", flush=True)

    if 0 in features:
        en0 = abs(obspy.signal.filter.envelope(st_x))
        envelope0 = np.array([])

    if 1 in features:
        st1 = st.copy()
        st1.filter('highpass',freq=5)
        am1 = st1.data
        en1 = abs(obspy.signal.filter.envelope(am1))
        envelope1 = np.array([])

    if 2 in features:
        st2 = st.copy()
        st2.filter('bandpass',freqmin=5,freqmax=10)
        am2 = st2.data
        en2 = abs(obspy.signal.filter.envelope(am2))
        envelope2 = np.array([])

    if 3 in features:
        st3 = st.copy()
        st3.filter('highpass',freq=10)
        am3 = st3.data
        en3 = abs(obspy.signal.filter.envelope(am3))
        envelope3 = np.array([])

    if 4 in features:
        freqmax_unfiltered = np.array([])

    if 5 in features:
        freq25th = np.array([])

    if 6 in features:
        freq50th = np.array([])

    if 7 in features:
        freq75th = np.array([])

    if 8 in features:
        kurtosis_time = np.array([])

    if 9 in features:
        kurtosis_freq = np.array([])

    if 10 in features:
        skewness_time = np.array([])

    if 11 in features:
        skewness_freq = np.array([])

    segleng=20*(1/st.stats.delta)
    nperseg=2**np.ceil(np.log2(segleng))
    nfft=4*nperseg
    ww = (1/(st.stats.delta))*60*window
    ll = ww*(1-overlap)
    
    print('\rStarting iterations-------------',end="", flush=True)

    for a in np.arange(0,(len(st_x))-ww,ll):
                
        times = np.append(times,st_t[int(a+(ww/2))]) # Times
        sub = st_x[int(a):int(a+ww)] # Subset unfiltered

        if 0 in features:
            envelope0 = np.append(envelope0,np.mean(en0[int(a):int(a+ww)]))

        if 1 in features:
            envelope1 = np.append(envelope1,np.mean(en1[int(a):int(a+ww)]))

        if 2 in features:
            envelope2 = np.append(envelope2,np.mean(en2[int(a):int(a+ww)]))

        if 3 in features:
            envelope3 = np.append(envelope3,np.mean(en3[int(a):int(a+ww)]))

        if (4 in features) or (5 in features) or (6 in features) or (7 in features):
            ff,pp = signal.welch(sub,fs=(1/st.stats.delta),window='hann',
                    nperseg=nperseg,noverlap=nperseg/2,nfft=nfft)
            if 4 in features:
                freqmax_unfiltered = np.append(freqmax_unfiltered,ff[np.argmax(pp)])
            csd=np.cumsum(pp)
            csd=csd-np.min(csd[1:])
            csd=csd/csd.max()
            if 5 in features:
                idx=np.argmin(np.abs(csd-.25))
                freq25th = np.append(freq25th,ff[idx])
            if 6 in features:
                idx=np.argmin(np.abs(csd-.50))
                freq50th = np.append(freq50th,ff[idx])
            if 7 in features:
                idx=np.argmin(np.abs(csd-.75))
                freq75th = np.append(freq75th,ff[idx])
        
        if 8 in features:
            kurtosis_time = np.append(kurtosis_time,stats.kurtosis(sub))

        if 9 in features:
            kurtosis_freq = np.append(kurtosis_freq,stats.kurtosis(pp))

        if 10 in features:
            skewness_time = np.append(skewness_time,stats.skew(sub))

        if 11 in features:
            skewness_freq = np.append(skewness_freq,stats.skew(pp))
        
        print(f'\r{100*((a+1)/(len(st_x)-ww+1)):3.2f}% done--------------------',
              end='',
              flush=True)

    print('\r100.00 % done-------------------\nFeatures extracted', end='', flush=True)
    
    data_frame = pd.DataFrame({'Times':times}) 

    if 12 in features:    
        cat = np.zeros(times.shape)
        cat[(times>=starttime) & (times<=endtime)] = 1
        data_frame[feature_labels[12]] = cat
      
    if 11 in features:
        data_frame.insert(loc=0,column=feature_labels[11],value=skewness_freq)
    if 10 in features:
        data_frame.insert(loc=0,column=feature_labels[10],value=skewness_time)
    if 9 in features:
        data_frame.insert(loc=0,column=feature_labels[9],value=kurtosis_freq)
    if 8 in features:
        data_frame.insert(loc=0,column=feature_labels[8],value=kurtosis_time)
    if 7 in features:
        data_frame.insert(loc=0,column=feature_labels[7],value=freq75th)
    if 6 in features:
        data_frame.insert(loc=0,column=feature_labels[6],value=freq50th)
    if 5 in features:
        data_frame.insert(loc=0,column=feature_labels[5],value=freq25th)
    if 4 in features:
        data_frame.insert(loc=0,column=feature_labels[4],value=freqmax_unfiltered)
    if 3 in features:
        data_frame.insert(loc=0,column=feature_labels[3],value=envelope3)
    if 2 in features:
        data_frame.insert(loc=0,column=feature_labels[2],value=envelope2)
    if 1 in features:
        data_frame.insert(loc=0,column=feature_labels[1],value=envelope1)
    if 0 in features:
        data_frame.insert(loc=0,column=feature_labels[0],value=envelope0)

    os.remove(fn)

    if keep:
        return data_frame, stored_stream
    else:
        return data_frame

#%% train_test_knn

def train_test_knn(data_frame,scale=True,neighbors=None,get_n=True,plot_n=True):
        
    X_train, X_test, y_train, y_test = train_test_split(data_frame.drop(['Times','Classification'],axis=1),data_frame['Classification'],test_size=0.50)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train,columns=data_frame.columns[:-2])
        X_test = pd.DataFrame(X_test,columns=data_frame.columns[:-2])

        if get_n:
            train = pd.concat([X_train, y_train], axis=1)

            k_range = range(1,int(len(train)*(4/5)))
            k_scores = []
            kf = KFold(n_splits=5)

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, data_frame.drop(['Times','Classification'],axis=1), data_frame['Classification'], cv=kf)
                k_scores.append(1-scores.mean())

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]

            if plot_n:
                fig = plt.figure(figsize=(5,3))
                plt.plot(k_range,k_scores)
                plt.xlabel('Value of K for KNN')
                plt.ylabel('Cross-Validated Error')
                plt.title('KNN Cross-Validation')
                plt.show()

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]
            neighbors=n

        model = KNeighborsClassifier(n_neighbors=int(neighbors))
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        report = classification_report(y_test,pred)
        conmat = confusion_matrix(y_test,pred)
        
        return model, scaler, report, conmat, neighbors
    
    else:
        if get_n:
            train = pd.concat([X_train, y_train], axis=1)

            k_range = range(1,int(len(train)*(4/5)))
            k_scores = []
            kf = KFold(n_splits=5)

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, data_frame.drop(['Times','Classification'],axis=1), data_frame['Classification'], cv=kf)
                k_scores.append(1-scores.mean())

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]

            if plot_n:
                fig = plt.figure(figsize=(5,3))
                plt.plot(k_range,k_scores)
                plt.xlabel('Value of K for KNN')
                plt.ylabel('Cross-Validated Error')
                plt.title('KNN Cross-Validation')
                plt.show()

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]
            neighbors=n

        model = KNeighborsClassifier(n_neighbors=int(neighbors))
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        report = classification_report(y_test,pred)
        conmat = confusion_matrix(y_test,pred)
        
        return model, report, conmat, neighbors
    
#%% predict_knn

def predict_knn(data_frame,model,scaler=[]):

    if 'Classification' in data_frame:
        drop = ['Times','Classification']
    else:
        drop = ['Times']
        
    dropped = data_frame[drop]
    data_frame = data_frame.drop(drop,axis=1)
    
    if isinstance(scaler,sklearn.preprocessing._data.StandardScaler):
        X_input = pd.DataFrame(scaler.transform(data_frame),columns=data_frame.columns[:5])
        prd = model.predict(X_input)
    else:
        prd = model.predict(data_frame)
        
    classified_data_frame = data_frame.join(dropped)
    classified_data_frame['Prediction'] = prd
        
    return classified_data_frame

#%% clean_detections

def clean_detections(data_frame,min_gap=20,min_sig=60,min_duration=20):

# Indices of all non-predictions
    zero_index = np.array(data_frame.index[data_frame['Prediction']==0].tolist())
    ones_index = np.array(data_frame.index[data_frame['Prediction']==1].tolist())

    if np.any(zero_index):

        # Time between samples
        tst = data_frame['Times'].iloc[1]-data_frame['Times'].iloc[0]

        pred1 = zero_index[:-1]
        pred2 = zero_index[1:]

        # Time between non-predictions in seconds
        pred3 = pred2 - pred1
        pred3 = pred3*tst

        # Find continuous predictions that are longer than min_duration
        cont = np.where(pred3>min_duration*60)[0]

    # Significant detections

    # If predictions start at index 0, include the first prediction
    if not(0 in zero_index):
        pair_0 = [0,zero_index[0]-1]

    # If predictions end at the last index, include the last prediction
        if not(len(data_frame['Prediction'])-1 in zero_index):
            sig_pairs = [[zero_index[i]+1,zero_index[i+1]-1] for i in cont[:-1]]
            pair_1 = [zero_index[-1]+1,len(data_frame['Prediction'])-1]
            sig_pairs.insert(0,pair_0)
            sig_pairs.append(pair_1)
        else:
            sig_pairs = [[zero_index[i]+1,zero_index[i+1]-1] for i in cont]
            sig_pairs.insert(0,pair_0)

    else:
        if not(len(data_frame['Prediction'])-1 in zero_index):
            sig_pairs = [[zero_index[i]+1,zero_index[i+1]-1] for i in cont[:-1]]
            pair_1 = [zero_index[-1]+1,len(data_frame['Prediction'])-1]
            sig_pairs.append(pair_1)
        else:
            sig_pairs = [[zero_index[i]+1,zero_index[i+1]-1] for i in cont]

    # If two detections are closer than min_sig, merge them
    for i in range(len(sig_pairs)-1):
        if ((sig_pairs[i+1][0]-sig_pairs[i][1])*tst)<min_sig*60*2:
            sig_pairs[i+1][0] = sig_pairs[i][0]
            sig_pairs[i][1] = sig_pairs[i+1][1]

    # Remove duplicates
    sig_pairs_x = []
    for i in sig_pairs:
        if i not in sig_pairs_x:
            sig_pairs_x.append(i)

    # Merge detections that are close to significant detections

    final_pairs = []

    for i in sig_pairs_x:
        bw_i = i[0]-1
        fw_i = i[1]+1
        bw_d = np.where(ones_index<bw_i)[0]
        fw_d = np.where(ones_index>fw_i)[0]

        if np.any(bw_d):
            while ((bw_i-ones_index[bw_d[-1]])*tst)<min_gap*60:
                bw_i = ones_index[bw_d[-1]]
                bw_d = np.where(ones_index<bw_i)[0]
                if not(np.any(bw_d)):
                    break

        if np.any(fw_d):
            while ((ones_index[fw_d[0]]-fw_i)*tst)<min_gap*60:
                fw_i = ones_index[fw_d[0]]
                fw_d = np.where(ones_index>fw_i)[0]
                if not(np.any(fw_d)):
                    break

        final_pairs.append([bw_i,fw_i])

    # Replace values

    detections = np.zeros(len(data_frame['Prediction']))

    for i in final_pairs:
        detections[i[0]:i[1]] = 1

    data_frame['Detection'] = detections

    return data_frame

#%% retrieve_dates

def retrieve_dates(data_frame,target='Detection'):

    det = np.array(data_frame.index[data_frame[target]==1].tolist())

    if np.any(det):

        det1 = det[:-1]
        det2 = det[1:]
        det3=det2-det1

        det_i1 = (np.where(abs(det3)>1)[0])
        det_i2 = [det[i] for i in det_i1]
        det_i3 = [det[i+1] for i in det_i1]

        if np.any(det_i2):
            det_0 = det[0]
            det_1 = det_i2
            det_0 = np.append(det_0,det_i3)
            det_1 = np.append(det_1,det[-1])
            starttimes = [UTCDateTime(data_frame['Times'].iloc[i]) for i in det_0]
            endtimes = [UTCDateTime(data_frame['Times'].iloc[i]) for i in det_1]
        else:
            det_0 = det[0]
            det_1 = det[-1]
            starttimes = [UTCDateTime(data_frame['Times'].iloc[det_0])]
            endtimes = [UTCDateTime(data_frame['Times'].iloc[det_1])]
        

        if det[0]==0:
            open_start = True
        else:
            open_start = False

        if det[-1]==len(data_frame[target]):
            open_end = True
        else:
            open_end = False

    else:
        starttimes = []
        endtimes = []
        open_start = False
        open_end = False

    return starttimes,endtimes,open_start,open_end

#%%

def FDSN_download_plot(client,network,station,location,channel,starttime,endtime,extended=0,decimate=[],vmin=-125,vmax=125):

    st = client.get_waveforms(network,
                              station,
                              location,
                              channel,
                              starttime=starttime-(extended*3600),
                              endtime=endtime+(extended*3600))

    st = st.merge(fill_value='interpolate').detrend('linear')
    st = st.select(component='Z')[0]

    if decimate:
        st.decimate(factor=decimate)
    
    st_x = st.data
    st_t = st.times(type='matplotlib')

    isamp = int(1/(st.stats.delta))
    inseg = isamp*10
    infft = 2048

    fig, ax = plt.subplots(2,1,sharex=True,figsize=(15,7))
    ax[0].plot(st_t,st_x,lw=0.5,c='k')
    ax[0].set_xlim(min(st_t),max(st_t))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax[0].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    ax[0].set_title(station)
    xspf,xspt,xsps=signal.spectrogram(st_x,fs=isamp,nperseg=inseg,nfft=infft,detrend=False)
    xspd = 20*np.log10(abs(xsps))
    ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
    im=ax[1].imshow(xspd,origin='lower',interpolation='nearest',aspect='auto',
                extent=[st_t[0],st_t[-1],xspf[0],xspf[-1]],
                cmap='jet',vmin=vmin,vmax=vmax)
    ax[1].set_ylabel('Hz')
    plt.tight_layout
    plt.show()

#%%

def SeisComP_download_plot(head='',
                           network='',
                           station='',
                           location='',
                           channel='',
                           starttime=[],
                           endtime=[],
                           extended=[],
                           decimate=[],
                           vmin=-125,
                           vmax=125):

    if ~extended:
        extended=0
    else:
        extended=extended

    if starttime and endtime:

        t1 = starttime-(extended*3600)
        t2 = endtime+(extended*3600)

        str_starttime = (f'{t1.year:04d}-'+
                        f'{t1.month:02d}-'+
                        f'{t1.day:02d}T'+
                        f'{t1.hour:02d}%3A'+
                        f'{t1.minute:02d}%3A'+
                        f'{t1.second:02d}')
        str_endtime = (f'{t2.year:04d}-'+
                    f'{t2.month:02d}-'+
                    f'{t2.day:02d}T'+
                    f'{t2.hour:02d}%3A'+
                    f'{t2.minute:02d}%3A'+
                    f'{t2.second:02d}')
    
    url1 = head
    url2 = 'starttime='+str_starttime+'&'
    url3 = 'endtime='+str_endtime+'&'
    url4 = 'network='+network+'&'
    url5 = 'station='+station+'&'
    url6 = 'location='+location+'&'
    url7 = 'channel='+channel+'&'
    url8 = 'nodata='+'404'

    url = url1

    if starttime:
        url+=url2
    if endtime:
        url+=url3
    if network != '':
        url+=url4
    if station != '':
        url+=url5
    if location != '':
        url+=url6
    if channel != '':
        url+=url7

    path = 'M:/FuegoLahar/Detection_2022/'+station+'_MSEED/'

    Path(path).mkdir(parents=True, exist_ok=True)

    fn = ('temp_stream.mseed')
    ff = requests.get(url)
    with open(path+'/'+fn, 'wb') as f:
        f.write(ff.content)

    st = obspy.read(path+'/'+fn)

    st = st.merge(fill_value='interpolate').detrend('linear')
    st = st.select(component='Z')[0]
    
    if decimate:
        st.decimate(factor=decimate)

    st_x = st.data
    st_t = st.times(type='matplotlib')

    isamp =int(1/(st.stats.delta))
    inseg = isamp*10
    infft = 2048

    fig, ax = plt.subplots(2,1,sharex=True,figsize=(15,7))
    ax[0].plot(st_t,st_x,lw=0.5,c='k')
    ax[0].set_xlim(min(st_t),max(st_t))
    ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
    ax[0].set_title(station)
    xspf,xspt,xsps=signal.spectrogram(st_x,fs=isamp,nperseg=inseg,nfft=infft,detrend=False)
    xspd = 20*np.log10(abs(xsps))
    ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    im=ax[1].imshow(xspd,origin='lower',interpolation='nearest',aspect='auto',
              extent=[st_t[0],st_t[-1],xspf[0],xspf[-1]],
              cmap='jet',vmin=vmin,vmax=vmax)
    ax[1].set_ylabel('Hz')
    plt.suptitle(station,y=0.95)
    plt.tight_layout
    plt.show()

    os.remove(path+'/'+fn)

#%% plot_detections

def plot_detections(data_frame,st,target='Detection',vmin=-125,vmax=125,save=False,save_path='',show=True,count_lim=[]):

    dt = retrieve_dates(data_frame,target=target)

    t0 = [x.matplotlib_date for x in dt[0]]
    t1 = [x.matplotlib_date for x in dt[1]]

    st = st.merge(fill_value='interpolate').detrend('linear')
    st = st.select(component='Z')[0]
    
    st_x = st.data
    st_t = st.times(type='matplotlib')

    isamp = int(1/(st.stats.delta))
    inseg = isamp*10
    infft = 2048

    fig, ax = plt.subplots(2,1,sharex=True,figsize=(15,7))
    ax[0].plot(st_t,st_x,lw=0.5,c='k')
    ax[0].set_xlim(min(st_t),max(st_t))
    if count_lim:
        ax[0].set_ylim(-count_lim,count_lim)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax[0].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    ax[0].set_title(st.stats.station)
    for i in range(len(t0)):
        ax[0].axvspan(t0[i],t1[i],color='red',alpha=0.25)
    xspf,xspt,xsps=signal.spectrogram(st_x,fs=isamp,nperseg=inseg,nfft=infft,detrend=False)
    xspd = 20*np.log10(abs(xsps))
    ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
    im=ax[1].imshow(xspd,origin='lower',interpolation='nearest',aspect='auto',
                extent=[st_t[0],st_t[-1],xspf[0],xspf[-1]],
                cmap='jet',vmin=vmin,vmax=vmax)
    for i in range(len(t0)):
        ax[1].axvspan(t0[i],t1[i],color='black',alpha=0.5)
    ax[1].set_ylabel('Hz')
    fig.suptitle('Detection between\n'+st.stats.starttime.strftime('%Y-%m-%d''T''%H:%M:%S')+
                 ' to '+st.stats.endtime.strftime('%Y-%m-%d''T''%H:%M:%S'))
    plt.tight_layout

    if show:
        plt.show()

    if save:
        if save_path == '':
            save_path = os.getcwd()
        plt.savefig(save_path+'/'+st.stats.station+'_'+st.stats.starttime.strftime('%Y%m%d%H%M%S')+'.png')
        plt.close()