## laharml (outdated README)

#### Required libraries
- os
- time
- numpy
- obspy
- scipy
- pandas
- sklearn
- requests
- pathlib
- matplotlib

#### Historic detections

To extract lahar detections from the historic record, use the corresponding `extract_from_record_FDSN.py` or `extract_from_record_SeisComP.py` script. The former works for any server that implements the FDSN web service definition through the Obspy package. The latter accesses the dataselect SeisComP platform, in this case at INSIVUMEH (VPN required). Keep the `laharml.py` module and the `training.txt` file in the same directory. The scripts are divided in three main cells: 1) package import, 2) user input, and 3) script. For now, modify only the variables in the second cell. Each cell is delimited by a line of pound symbols (#).

- **start_date:** (`str`) start datetime for the period of interest the script will extract detections from. Use the `"%Y-%m-%d"` or the `"%Y-%m-%d""T""%H:%M:%S"` format.
- **end_date:** end datetime for the period of interest the script will extract detections from. Use the `"%Y-%m-%d"` or the `"%Y-%m-%d""T""%H:%M:%S"` format.
- **client_name:** (`str`; in `extract_from_record_FDSN.py` only) FDSN client name.
- **client_username:** (`str`; in `extract_from_record_FDSN.py` only) specify username if credentials are required, otherwise leave as `""`.
- **client_password:** (`str`; in `extract_from_record_FDSN.py` only) specify password if credentials are required, otherwise leave as `""`.
- **server_address:** (`str`; in `extract_from_record_SeisComP.py` only) address to the FDSNWS with a SeisComP backend.
- **network:** (`str`) seismic network. Uses Unix style wildcard (`"*"`, `?`, etc).
- **station:** (`str`) station name. Uses Unix style wildcard (`"*"`, `?`, etc).
- **location:** (`str`) location code. Uses Unix style wildcard (`"*"`, `?`, etc).
- **channel:** (`str`) seismic network. Uses Unix style wildcard (`"*"`, `?`, etc).
- **features:** (`list`) Array of codes of desired features to extract.
    - 0: Unfiltered envelope
    - 1: Envelope of filtered signal (above 5 Hz)
    - 2: Envelope of filtered signal (between 5 and 10 Hz)
    - 3: Envelope of filtered signal (above 10 Hz)
    - 4: Maximum power frequency
    - 5: 25th percentile frequency
    - 6: 50th percentile frequency
    - 7: 75th percentile frequency
    - 8: Kurtosis of amplitudes
    - 9: Kurtosis of frequencies
    - 10: Skewness of amplitudes
    - 11: Skewness of frequencies
    - 12: Classification as lahar signal for use when training and fitting ML models if start and end times of lahar signal are known.
- **output_folder_path:** (`str`) changes directory of output files. To use current directory use `""`.
- **show_detections:** (`bool`) shows plots of detections during script run.
- **save_log:** (`bool`) saves raw detections.
- **save_detections:** (`bool`) saves final detection dates.

The `training.txt` file specifies training events for a list of stations and it should be updated accordingly.

The `extract_from_record_FDSN.py` and `extract_from_record_SeisComP.py` scripts require the `laharml` module. To modify the behavior of the detection extraction scripts, change the keyboard arguments of the functions in the `laharml` module.

By default, `extract_from_record_FDSN.py` and `extract_from_record_SeisComP.py` are set up to generate predictions for the second half of May 2022 in FEC1 and FG16 respectively (user input cell variables).

#### Important functions in extraction scripts

**Line 94 (also Line 125):** `FDSN_download_extract()` and `SeisComP_download_extract()`

Requests and temporarily downloads an obspy.Stream object for the provided arguments. Extracts the features requested in the `features` variable (user input cell). Use `keep=True` to store a variable with the downloaded obspy.Stream object. Use `plot=True` to show a plot of the downloaded seismogram in the time and frequency domains. Returns a pandas.DataFrame object with the extracted features. If `keep=True`, it also returns a obspy.Stream object. Definition in Line 23 (FDSN) and Line 262 (SeisComP) of the `laharml.py` module.

**Line 97:** `train_test_knn()`

Reads the pandas.DataFrame output object from the previous function and splits into training and testing datasets. Tests the performance of the classification using the KNN algorithm from scikit-leanr. Use `get_n=True` to perform a cross-validation using the training dataset and retrieve a number of neighbors. Use `plot_n=True` to show a plot of the cross-validation results. Returns a sklearn.KNeighborsClassifier object, a string with the classification report, a numpy.ndarray() object with the confusion matrix, and an integer with the number of neighbors used for the model. If `scale=True`, it also returns a sklearn.Scaler object. Definition in Line 549 of the `laharml.py` module.

**Line 126:** `predict_knn()`

Reads an unclassified pandas.DataFrame of extracted features and uses the estimated sklearn.KNeighborsClassifier model (and sklearn.Scaler object if applicable) to generate predictions of a given variable based on the features in the `features` variable (user input cell). Returns a classified pandas.DataFrame, similar to the unclassified pandas.DataFrame input with an added 'Detection' column that specifies the KNN model predictions. Definition in Line 633 of the `laharml.py` module.

**Line 127:** `clean_detections()`

Reads the 'Detection' column in the classified pandas.DataFrame from the previous function. It removes potential false positives and merges continuous detections reducing potential false negatives. Returns a pandas.DataFrame where the 'Detection' column has been updated with reduced erronous predictions. Definition in Line 656 of the `laharml.py` module.

**Line 128:** `retrieve_dates()`

Reads through the 'Detection' column in classified pandas.DataFrame and extracts the start and end datetimes of continous predictions. Returns a numpy.ndarray of starting times, a numpy.ndarray of ending times, a numpy.ndarray of booleans specifying if the startin times matches the first datetime element where the detected signal could potentially start before the start time of the original obspy.Stream object (`True`), and a numpy.ndarray of booleans specifying if the ending time matches the last datetime element where the detected signal could potentially end after the end time of the original obspy.Stream object (`True`). Definition in Line 752 of the `laharml.py` module.

**Line 134:** `plot_detections()`

Uses the stored obspy.Stream object (Line 125) to plot the seismogram in the time and frequency domains highlighting the periods where detections have occurred. For this, `target="Detection"`. Returns a matplotlib.Figure with the plots.
