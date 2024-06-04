import numpy as np
import pandas as pd
import math
import pywt

def Dissolves_array_reshape (array):
    '''
    reshape(-1, ndarray.shape[-1]): This method reshapes the array into a 2D array. 
    The -1 argument in the first dimension means that NumPy will automatically determine the number of rows based on total elements can contain in new matrix 
    so this mean rearrangement matrix elements to fit new shape

    dissolves the array into 2d array this exactly mean keep last array save  [*] for any change and merge others [ [[*],[*]] , [[*],[*]] ] => [[*],[*],[*],[*]]
    '''
    arrayMatrix= np.array(array)
    arrayMatrix= arrayMatrix.reshape(-1,arrayMatrix.shape[-1])# dissolves the array into 2d array this exactiy mean keep last array save  [*] for any change and merge others [ [[*],[*]] , [[*],[*]] ] => [[*],[*],[*],[*]]
    arrayMatrix = np.squeeze(arrayMatrix) # remove the single-dimensional entries from the shape of an array. not need but if no one`s in shape tuble no effect
    return arrayMatrix.tolist()




def features_estimation(signal, channel_name, fs, frame, step):
    """
    Compute time, frequency and time-frequency features from signal.
    :param signal: numpy array signal.
    :param channel_name: string variable with the EMG channel name in analysis.
    :param fs: int variable with the sampling frequency used to acquire the signal
    :param frame: sliding window size
    :param step: sliding window step size

    :return: total_feature_matrix -- python Dataframe with .
    :return: features_names -- python list with  total_feature_matrixpd, features_names ,time_matrix,total_feature_matrix_np

    """

    features_names = ['VAR', 'RMS', 'IEMG','SSI', 'MAV', 'LOG', 'WL', 'ACC','M2','DVARV', 'DASDV', 'ZC', 'WAMP', 'MYOP','IE', "FR", "MNP", "TP",
                      "MNF", "MDF", "PKF", "WENT"]

    time_matrix = time_features_estimation(signal, frame, step)
    frequency_matrix = frequency_features_estimation(signal, fs, frame, step)
    time_frequency_matrix = time_frequency_features_estimation(signal, frame, step)
    total_feature_matrixpd = pd.DataFrame(np.column_stack((time_matrix, frequency_matrix, time_frequency_matrix)).T,
                                        index=features_names)
    total_feature_matrix_np =np.column_stack((time_matrix, frequency_matrix, time_frequency_matrix))


 

    return total_feature_matrixpd, features_names ,time_matrix,total_feature_matrix_np


def time_features_estimation(signal, frame, step):
    """
    Compute time features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size.
    :param step: sliding window step size.

    :return: time_features_matrix: narray matrix with the time features stacked by columns.
    """

    variance = []
    rms = []
    iemg = []
    ssi= []
    mav = []
    log_detector = []
    wl = []
    aac = []
    m2 = []
    dvarv = []
    dasdv = []
    zc = []
    wamp = []
    myop = []
    ie=[]
    #this array will append each windows measures  value after make calculation in it so each index represent for one window measure 
    th = np.mean(signal) + 3 * np.std(signal)

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]

        variance.append(np.var(x))
        rms.append(np.sqrt(np.mean(x ** 2)))
        iemg.append(np.sum(abs(x)))  # Integral
        ssi.append(np.sum(x ** 2))  # Signal Strength Index
        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
        log_detector.append(np.exp(np.sum(np.log10(np.absolute(x))) / frame))
        wl.append(np.sum(abs(np.diff(x))))  # Wavelength
        aac.append(np.sum(abs(np.diff(x))) / (frame-1))  # Average Amplitude Change
        m2.append(np.sum(np.diff(x) ** 2) )  # Second moment
        dvarv.append(np.sum(np.diff(x)**2)/(frame-2))  # Difference variance version
        dasdv.append(
            math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value
        zc.append(zcruce(x, th))  # Zero-Crossing
        wamp.append(wilson_amplitude(x, th))  # Willison amplitude
        myop.append(myopulse(x, th))  # Myopulse percentage rate
        ie.append(np.sum(np.exp (x)))  # integral of the exponential of the signal

    time_features_matrix = np.column_stack((variance, rms, iemg  ,ssi , mav, log_detector, wl, aac,m2,dvarv, dasdv, zc, wamp, myop,ie))
    return time_features_matrix


def frequency_features_estimation(signal, fs, frame, step):
    """
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param fs: sampling frequency of the signal.
    :param frame: sliding window size
    :param step: sliding window step size
    
    ex...
    The signal array contains integers from 1 to 12.
    frame is set to 3, so each window will contain 3 elements.
    step is set to 2, so the windows will slide by 2 elements at a time.

    :return: frequency_features_matrix: narray matrix with the frequency features stacked by columns.
    """

    fr = []
    mnp = []
    tot = []
    mnf = []
    mdf = []
    pkf = []

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        fr.append(frequency_ratio(frequency, power))  # Frequency ratio
        mnp.append(np.sum(power) / len(power))  # Mean power
        tot.append(np.sum(power))  # Total power
        mnf.append(mean_freq(frequency, power))  # Mean frequency
        mdf.append(median_freq(frequency, power))  # Median frequency
        pkf.append(frequency[power.argmax()])  # Peak frequency

    frequency_features_matrix = np.column_stack((fr, mnp, tot, mnf, mdf, pkf))

    return frequency_features_matrix


def time_frequency_features_estimation(signal, frame, step):
    """
    Compute time-frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: h_wavelet: list
    """
    h_wavelet = []

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]

        E_a, E = wavelet_energy(x, 'db2', 4)
        E.insert(0, E_a)
        E = np.asarray(E) / 100

        h_wavelet.append(-np.sum(E * np.log2(E)))

    return h_wavelet


def wilson_amplitude(signal, th):
    x = abs(np.diff(signal))
    umbral = x >= th
    return np.sum(umbral)


def myopulse(signal, th):
    '''Myopulse percentage rate is the percentage of the signal that is above a certain threshold. 
    in python if compare between array and number it will return array of boolean values so will compare each element in the array with the number '''
    umbral = signal >= th
    return np.sum(umbral) / len(signal)


def spectrum(signal, fs):
    m = len(signal)
    n = next_power_of_2(m)
    y = np.fft.fft(signal, n)
    yh = y[0:int(n / 2 - 1)]
    fh = (fs / n) * np.arange(0, n / 2 - 1, 1)
    power = np.real(yh * np.conj(yh) / n)

    return fh, power


def frequency_ratio(frequency, power):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / UHC


def shannon(x):
    N = len(x)
    nb = 19
    hist, bin_edges = np.histogram(x, bins=nb)
    counts = hist / N
    nz = np.nonzero(counts)

    return np.sum(counts[nz] * np.log(counts[nz]) / np.log(2))


def zcruce(X, th):
    th = 0
    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1
    return cruce


def mean_freq(frequency, power):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / den


def median_freq(frequency, power):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / power_total
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]


def wavelet_energy(x, mother, nivel):
    coeffs = pywt.wavedecn(x, wavelet=mother, level=nivel)
    arr, _ = pywt.coeffs_to_array(coeffs)
    Et = np.sum(arr ** 2)
    cA = coeffs[0]
    Ea = 100 * np.sum(cA ** 2) / Et
    Ed = []

    for k in range(1, len(coeffs)):
        cD = list(coeffs[k].values())
        cD = np.asarray(cD)
        Ed.append(100 * np.sum(cD ** 2) / Et)

    return Ea, Ed


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def med_freq(f, P):
    Ptot = np.sum(P) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += P[i]
        errel = (Ptot - temp) / Ptot
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return f[i]




def extract_features_toTensorflow(tablePanda) :
    """
    reshape from (feature ,window) to (window, features) 2d shape
    Extract features from the tablePanda and return a numpy list  with the all  features for each window shape(window, features)
    :param tablePanda: table with the data
    :return: numpy list  with the features for each window shape(window, features)
    
    :explain: table shape is (rows "features", columns "windows") reshape it to (windows ,  features)
    """
    tableShape = tablePanda.shape
   
    featuresMatrix = []
    for i in range(tableShape[1]):
        signal = tablePanda.iloc[:, i].values.tolist()
        
        featuresMatrix.append(signal)
   
     
    return featuresMatrix


def Handel_timeD_featuresEngeenring_withReshape (filter_emgMatrix, channels_name, fs, frame, step,extraction_features=[]) :
    '''
    this function for handle time domain feature extraction from emg signal for return matrix have shape (n_window, n_feature)
    let`s break down idea to first i get filter emg matrix shape (trials.size*samples.size  , channels.size )  so this 2d matrix
    so to call features_estimation function we need to extract each channel to feed through function call by transpose filter matrix
    after feature extraction return matrix  total_feature_matrix_channel1 @type panda data frame (feature ,window), features_names_channel2 @type row vector string  ,time_matrix shape (windows, features)

    so why need time domain ? because this matrix generate  depends on paper study

    @para: filter_emgMatrix : @type 2d numpy.array().shape(trials.size*samples.size  , channels.size)
    @para :channels_name : list|numpy.array() vector of string
    @fs :sample frequency
    @extraction_features : list contain element need extraction don`t pass value or pass empty list  automatic code extract time_featureName_specialForStudies
    @frame :frame length this window size mean number of samples for each window generated
    @step : this step for frame
    *for frame and step range(frame, signal.size, step) signal.size this for each channel
    this code range(frame, signal.size, step) this for generate list this list using in foreach loop
    example
    for i in range(frame,signal.size,step): => for i in [frame,frame+step , ... , signale.size ]:
          x = signal[i - frame:i] create window

    all extraction feature ['VAR', 'RMS', 'IEMG','SSI', 'MAV', 'LOG', 'WL', 'ACC','M2','DVARV', 'DASDV', 'ZC', 'WAMP', 'MYOP','IE', "FR", "MNP", "TP",
                      "MNF", "MDF", "PKF", "WENT"]

    @return : 2d numpy array shape (channels * n_window, n_feature)( total_feature_estimation, total_feature_estimation_time ,extraction_features_name)
    total_feature_estimation this matrix get exact feature matrix  need
    '''
    time_featureName_specialForStudies  =['VAR', 'RMS', 'IEMG','SSI', 'MAV', 'WL','ACC','M2','DVARV', 'DASDV', 'WAMP','MYOP','IE'] #, 'MYOP' remove because some point make overlape feature for gesture 16 17
    if not extraction_features:
        extraction_features=time_featureName_specialForStudies
    total_feature_estimation_time =[]
    total_feature_estimation=[]
    counter=0
    filter_emgMatrix = filter_emgMatrix.T
    for channel in filter_emgMatrix:
        total_feature_matrix, _ , time_matrix,_ = features_estimation(channel,channels_name[counter], fs,frame, step)
        counter += 1
        total_feature_estimation_time.append(time_matrix)
        total_feature_matrix_helper = total_feature_matrix.loc[extraction_features].T.to_numpy()
        total_feature_matrix_helper= total_feature_matrix_helper.tolist()
        total_feature_estimation.append(total_feature_matrix_helper)#matrix start shape channel ,window , features

    # Reshapeing  by function in classification utility  convert from 3d to 2d (channels,windowframes , features )  => (channels * windowframes , features )
    total_feature_estimation_time = Dissolves_array_reshape (total_feature_estimation_time)
    total_feature_estimation= Dissolves_array_reshape(total_feature_estimation)

    extraction_features_name=extraction_features
    return  total_feature_estimation, total_feature_estimation_time,extraction_features_name


