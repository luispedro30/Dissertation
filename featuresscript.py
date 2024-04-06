import numpy as np
import pandas as pd
import pyaudio
import librosa
import pywt
import wave 
import joblib, nolds
from scipy.stats import skew,kurtosis,entropy
from scipy.signal import find_peaks

def calculate_app_tkeo_mean(audio_signal, num_segments=10):
    segment_size = len(audio_signal) // num_segments
    app_tkeo_mean_values = []

    for i in range(num_segments):
        start_index = i * segment_size
        end_index = start_index + segment_size

        # Extract the segment of the signal
        segment = audio_signal[start_index:end_index]

        # Calculate the Teager-Kaiser energy operator (TKEO) mean for the segment
        tkeo = segment[1:-1] ** 2 - segment[:-2] * segment[2:]
        app_tkeo_mean = np.mean(tkeo)

        # Append the TKEO mean to the list of values
        app_tkeo_mean_values.append(app_tkeo_mean)
        
    return app_tkeo_mean_values


def calculate_app_tkeo_std(audio_signal, num_segments=10):
    segment_size = len(audio_signal) // num_segments
    app_tkeo_std_values = []

    for i in range(num_segments):
        start_index = i * segment_size
        end_index = start_index + segment_size

        # Extract the segment of the signal
        segment = audio_signal[start_index:end_index]

        # Calculate the Teager-Kaiser energy operator (TKEO) mean for the segment
        tkeo = segment[1:-1] ** 2 - segment[:-2] * segment[2:]
        app_tkeo_std = np.std(tkeo)

        # Append the TKEO mean to the list of values
        app_tkeo_std_values.append(app_tkeo_std)
        
    return app_tkeo_std_values

def compute_ppe(audio_signal, sample_rate):
    # Compute the autocorrelation of the audio signal
    autocorr = librosa.autocorrelate(audio_signal)

    # Remove the first element (which is the autocorrelation at lag 0)
    autocorr = autocorr[1:]

    # Compute the normalized autocorrelation
    norm_autocorr = autocorr / np.max(autocorr)

    # Compute the Pitch Period Entropy (PPE)
    ppe = -np.sum(norm_autocorr * np.log(np.maximum(norm_autocorr, np.finfo(float).eps)))

    return ppe

def compute_jitter(audio_signal):
    diff = np.diff(audio_signal)  # Compute differences between consecutive samples
    jitter = np.mean(np.abs(diff))  # Compute mean absolute difference
    return jitter

def compute_shimmer_features(audio_signal, sample_rate):
    # Compute peak amplitudes
    peaks, _ = find_peaks(audio_signal)
    peak_amplitudes = audio_signal[peaks]

    # Compute differences between consecutive peak amplitudes
    peak_diffs = np.diff(peak_amplitudes)

    # Compute local shimmer
    loc_shimmer = np.mean(np.abs(peak_diffs))

    # Compute local shimmer in decibels
    loc_db_shimmer = np.mean(20 * np.log10(np.abs(peak_diffs) + 1e-9))  # Adding a small constant value

    # Compute APQ for 3 periods
    apq3_shimmer = np.mean(np.abs(np.diff(peak_diffs[:3])))

    # Compute APQ for 5 periods
    apq5_shimmer = np.mean(np.abs(np.diff(peak_diffs[:5])))

    # Compute APQ for 11 periods
    apq11_shimmer = np.mean(np.abs(np.diff(peak_diffs[:11])))

    # Compute DDA shimmer
    dda_shimmer = np.mean(np.abs(np.diff(peak_diffs[1::2]))) - np.mean(np.abs(np.diff(peak_diffs[::2])))

    return loc_shimmer, loc_db_shimmer, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer

def compute_features_vocal_fold(file_path):
    # Load the audio file
    signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
    features = {}

    # Compute GQ_prc5_95
    q95, q5 = np.percentile(signal, [95, 5])
    gq_prc5_95 = q95 - q5
    features['GQ_prc5_95'] = gq_prc5_95

    # Compute GQ_std_cycle_open and GQ_std_cycle_closed
    # These are just placeholders, you need to implement the actual computation based on your requirements
    gq_std_cycle_open = np.std(signal)
    gq_std_cycle_closed = np.std(signal)
    features['GQ_std_cycle_open'] = gq_std_cycle_open
    features['GQ_std_cycle_closed'] = gq_std_cycle_closed

    # Compute GNE_mean and GNE_std
    gne_mean = np.mean(np.abs(np.diff(signal)))
    gne_std = np.std(np.abs(np.diff(signal)))
    features['GNE_mean'] = gne_mean
    features['GNE_std'] = gne_std

    # Compute GNE_SNR_TKEO and GNE_SNR_SEO
    gne_snr_tkeo = np.mean(signal ** 2) / np.mean(np.diff(signal) ** 2)
    gne_snr_seo = np.mean(np.abs(signal)) / np.mean(np.abs(np.diff(signal)))
    features['GNE_SNR_TKEO'] = gne_snr_tkeo
    features['GNE_SNR_SEO'] = gne_snr_seo

    # Compute GNE_NSR_TKEO and GNE_NSR_SEO
    # These are just placeholders, you need to implement the actual computation based on your requirements
    gne_nsr_tkeo = np.mean(signal) / np.mean(np.diff(signal))
    gne_nsr_seo = np.mean(signal) / np.mean(np.diff(signal))
    features['GNE_NSR_TKEO'] = gne_nsr_tkeo
    features['GNE_NSR_SEO'] = gne_nsr_seo

    # Compute VFER_mean and VFER_std
    vfer_mean = np.mean(np.diff(signal))
    vfer_std = np.std(np.diff(signal))
    features['VFER_mean'] = vfer_mean
    features['VFER_std'] = vfer_std

    # Compute VFER_entropy
    vfer_entropy = entropy(signal)
    features['VFER_entropy'] = vfer_entropy

    # Compute VFER_SNR_TKEO and VFER_SNR_SEO
    # These are just placeholders, you need to implement the actual computation based on your requirements
    vfer_snr_tkeo = np.mean(signal ** 2) / np.mean(np.diff(signal) ** 2)
    vfer_snr_seo = np.mean(np.abs(signal)) / np.mean(np.abs(np.diff(signal)))
    features['VFER_SNR_TKEO'] = vfer_snr_tkeo
    features['VFER_SNR_SEO'] = vfer_snr_seo

    # Compute VFER_NSR_TKEO and VFER_NSR_SEO
    # These are just placeholders, you need to implement the actual computation based on your requirements
    vfer_nsr_tkeo = np.mean(signal) / np.mean(np.diff(signal))
    vfer_nsr_seo = np.mean(signal) / np.mean(np.diff(signal))
    features['VFER_NSR_TKEO'] = vfer_nsr_tkeo
    features['VFER_NSR_SEO'] = vfer_nsr_seo

    # Compute IMF_SNR_SEO, IMF_SNR_TKEO, and IMF_SNR_entropy
    # These are just placeholders, you need to implement the actual computation based on your requirements
    imf_snr_seo = np.mean(np.abs(signal)) / np.mean(np.abs(np.diff(signal)))
    imf_snr_tkeo = np.mean(signal ** 2) / np.mean(np.diff(signal) ** 2)
    imf_snr_entropy = entropy(signal)
    features['IMF_SNR_SEO'] = imf_snr_seo
    features['IMF_SNR_TKEO'] = imf_snr_tkeo
    features['IMF_SNR_entropy'] = imf_snr_entropy

    # Compute IMF_NSR_SEO, IMF_NSR_TKEO, and IMF_NSR_entropy
    # These are just placeholders, you need to implement the actual computation based on your requirements
    imf_nsr_seo = np.mean(signal) / np.mean(np.abs(np.diff(signal)))
    imf_nsr_tkeo = np.mean(signal) / np.mean(np.diff(signal))
    imf_nsr_entropy = entropy(signal)
    features['IMF_NSR_SEO'] = imf_nsr_seo
    features['IMF_NSR_TKEO'] = imf_nsr_tkeo
    features['IMF_NSR_entropy'] = imf_nsr_entropy

    return features

def compute_audio_features_mfccs(file_path):
    # Load the audio file
    audio_signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
    features = {}

    print("still")
    # Compute mean_Log_energy
    mean_log_energy = np.mean(np.log(np.abs(audio_signal) ** 2 + 1e-9))  # Adding a small constant
    features['mean_Log_energy'] = mean_log_energy

    # Compute mean and standard deviation of MFCC coefficients (0th to 12th)
    mfcc_coeffs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    for i in range(13):
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(i+1, 'th')  # Get the appropriate suffix
        features[f'mean_MFCC_{i+1}{suffix}_coef'] = np.mean(mfcc_coeffs[i])
        features[f'std_MFCC_{i+1}{suffix}_coef'] = np.std(mfcc_coeffs[i])

    # Compute mean and standard deviation of delta log energy and delta coefficients
    rms_energy = np.sqrt(np.mean(audio_signal ** 2))
    features['rms_energy'] = rms_energy
    delta_log_energy = librosa.feature.delta(np.log(np.abs(audio_signal)+ 1e-9))
    delta_mfcc = librosa.feature.delta(mfcc_coeffs)
    for i in range(13):
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(i+1, 'th')  # Get the appropriate suffix
        features[f'mean_delta_log_energy'] = np.mean(delta_log_energy[i])
        features[f'std_delta_log_energy'] = np.std(delta_log_energy[i])
        features[f'mean_{i+1}{suffix}_delta'] = np.mean(delta_mfcc[i])
        features[f'std_{i+1}{suffix}_delta'] = np.std(delta_mfcc[i])

    # Compute mean and standard deviation of delta-delta log energy and delta-delta coefficients
    delta_delta_log_energy = librosa.feature.delta(delta_log_energy)
    delta_delta_mfcc = librosa.feature.delta(delta_mfcc)
    for i in range(13):
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(i+1, 'th')  # Get the appropriate suffix
        features[f'mean_delta_delta_log_energy'] = np.mean(delta_delta_log_energy[i])
        features[f'std_delta_delta_log_energy'] = np.std(delta_delta_log_energy[i])
        features[f'mean_{i+1}{suffix}_delta_delta'] = np.mean(delta_delta_mfcc[i])
        features[f'std_{i+1}{suffix}_delta_delta'] = np.std(delta_delta_mfcc[i])

    return features


def calculate_features_wavelet(file_path):
    # Load the audio file
    audio_signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
    
    features = {}
    
    app_tkeo_mean_values = calculate_app_tkeo_mean(audio_signal)
    app_tkeo_std_values = calculate_app_tkeo_std(audio_signal)
    
    # Energy (Ea)
    energy = np.sum(audio_signal ** 2) / len(audio_signal)
    features['Ea'] = energy

    # Delta Coefficients (Ed_1_coef, Ed_2_coef, ..., Ed_10_coef)
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    for i in range(min(10, delta_mfcc.shape[0])):
        features[f'Ed_{i+1}_coef'] = np.mean(delta_mfcc[i])

    # Shannon Entropy (det_entropy_shannon_1_coef, ..., det_entropy_shannon_10_coef)
    entropy_bw = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate)
    for i in range(min(10, entropy_bw.shape[1])):
        features[f'det_entropy_shannon_{i+1}_coef'] = np.mean(entropy_bw[:, i])

    # Additional features
    envelope = np.abs(librosa.core.stft(audio_signal))

    # Take the logarithm of the envelope
    log_envelope = np.log1p(envelope)

    # Log Entropy (det_entropy_log_1_coef, ..., det_entropy_log_10_coef)
    for i in range(10):
        entropy_log = entropy(log_envelope, axis=0, base=2)
        features[f'det_entropy_log_{i+1}_coef'] = np.mean(entropy_log)

    # TKEO Mean (det_TKEO_mean_1_coef, ..., det_TKEO_mean_10_coef)
    tkeo = librosa.onset.onset_strength(y=audio_signal, sr=sample_rate)
    for i in range(10):
        features[f'det_TKEO_mean_{i+1}_coef'] = np.mean(tkeo)

    # TKEO Standard Deviation (det_TKEO_std_1_coef, ..., det_TKEO_std_10_coef)
    for i in range(10):
        features[f'det_TKEO_std_{i+1}_coef'] = np.std(tkeo)

    app_entropy = librosa.feature.spectral_flatness(y=audio_signal)
    num_frames = app_entropy.shape[1]  # Get the number of frames
    for i in range(min(10, num_frames)):
        features[f'app_entropy_shannon_{i+1}_coef'] = np.mean(app_entropy[:, i] ** 2)
        
    # App Entropy (app_entropy_log_1_coef, ..., app_entropy_log_10_coef)
    app_entropy = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate, centroid=None)
    num_frames = app_entropy.shape[1]  # Get the number of frames
    for i in range(min(10, num_frames)):
        features[f'app_entropy_log_{i+1}_coef'] = np.mean(app_entropy[:, i] ** 2)

    # LT TKEO Mean (det_LT_TKEO_mean_1_coef, ..., det_LT_TKEO_mean_10_coef)
    lt_tkeo = librosa.feature.tempogram(y=audio_signal, sr=sample_rate)
    
    num_frames = lt_tkeo.shape[1]  # Get the number of frames
    for i in range(min(10, num_frames)):
        features[f'det_LT_TKEO_mean_{i+1}_coef'] = np.mean(lt_tkeo[:, i])
    
    # App TKEO Mean (app_TKEO_mean_1_coef, ..., app_TKEO_mean_10_coef)
    #num_frames = len(app_tkeo_mean)
    #for i in range(min(10, num_frames)):
    for i in range(10):
        features[f'app_det_TKEO_mean_{i+1}_coef'] = app_tkeo_mean_values[i]

    # App TKEO Standard Deviation (app_TKEO_std_1_coef, ..., app_TKEO_std_10_coef)
    #num_frames = len(app_tkeo_std)
    #for i in range(min(10, num_frames)):
    for i in range(10):
        features[f'app_TKEO_std_{i+1}_coef'] = app_tkeo_std_values[i]


    # Additional features based on Ea
    # Ea2
    energy_squared = energy ** 2
    features['Ea2'] = energy_squared

    # Additional features based on Ed_2_coef
    # Ed2_1_coef, Ed2_2_coef, ..., Ed2_10_coef
    for i in range(10):
        features[f'Ed2_{i+1}_coef'] = np.mean(delta_mfcc[i] ** 2)

    # Additional features based on det_entropy_shannon_1_coef
    # det_LT_entropy_shannon_1_coef, ..., det_LT_entropy_shannon_10_coef
    for i in range(10):
        features[f'det_LT_entropy_shannon_{i+1}_coef'] = np.mean(entropy_bw[:, i] ** 2)

    # Additional features based on det_LT_entropy_log_1_coef
    # det_LT_entropy_log_1_coef, ..., det_LT_entropy_log_10_coef
    for i in range(10):
        features[f'det_LT_entropy_log_{i+1}_coef'] = np.mean(entropy_log ** 2)

    for i in range(10):
        features[f'det_LT_TKEO_mean_{i+1}_coef'] = np.mean(lt_tkeo[i])
        
    for i in range(10):
        features[f'det_LT_TKEO_std_{i+1}_coef'] = np.std(lt_tkeo[i])

    # Additional features based on app_entropy_shannon_1_coef
    # app_LT_entropy_shannon_1_coef, ..., app_LT_entropy_shannon_10_coef
    for i in range(min(10, app_entropy.shape[1])):
        features[f'app_LT_entropy_shannon_{i+1}_coef'] = np.mean(app_entropy[:, i] ** 2)

    # Additional features based on app_entropy_log_1_coef
    # app_LT_entropy_log_1_coef, ..., app_LT_entropy_log_10_coef
    for i in range(min(10, app_entropy.shape[1])):
        features[f'app_LT_entropy_log_{i+1}_coef'] = np.mean(app_entropy[:, i] ** 2)

    # Additional features based on app_det_TKEO_mean_1_coef
    # app_LT_TKEO_mean_1_coef, ..., app_LT_TKEO_mean_10_coef
    for i in range(10):
        features[f'app_LT_TKEO_mean_{i+1}_coef'] = np.mean(tkeo ** 2)

    # Additional features based on app_TKEO_std_1_coef
    # app_TKEO_std_1_coef, ..., app_TKEO_std_10_coef
    for i in range(10):
        features[f'app_LT_TKEO_std_{i+1}_coef'] = np.std(tkeo ** 2)

    return features

def shannon_entropy(signal):
    # Compute histogram of the signal
    hist, _ = np.histogram(signal, bins='auto', density=True)

    # Calculate Shannon entropy
    entropy_value = -np.sum(hist * np.log2(hist + np.finfo(float).eps))  # Add epsilon for numerical stability

    return entropy_value

def calculate_tqwt_features(file_path):
    # Load the audio file
    audio_signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
    
    features = {}
    
    # Calculate TQWT coefficients using Stationary Wavelet Transform
    tqwt_coeffs = pywt.swt(audio_signal, 'db1', level=5)

    # Calculate TQWT energy coefficients
    for i in range(36):
        if i < len(tqwt_coeffs):
            features[f'tqwt_energy_dec_{i+1}'] = np.sum(tqwt_coeffs[i])
            features[f'tqwt_entropy_shannon_dec_{i+1}'] = shannon_entropy(tqwt_coeffs[i])
            features[f'tqwt_entropy_log_dec_{i+1}'] = np.mean(librosa.feature.spectral_centroid(y=np.array(tqwt_coeffs[i]), sr=sample_rate))
            features[f'tqwt_TKEO_mean_dec_{i+1}'] = np.mean(tqwt_coeffs[i])
            features[f'tqwt_TKEO_std_dec_{i+1}'] = np.std(tqwt_coeffs[i])
            features[f'tqwt_medianValue_dec_{i+1}'] = np.median(tqwt_coeffs[i])
            features[f'tqwt_meanValue_dec_{i+1}'] = np.mean(tqwt_coeffs[i])
            features[f'tqwt_stdValue_dec_{i+1}'] = np.std(tqwt_coeffs[i])
            features[f'tqwt_minValue_dec_{i+1}'] = np.min(tqwt_coeffs[i])
            features[f'tqwt_maxValue_dec_{i+1}'] = np.max(tqwt_coeffs[i])
            features[f'tqwt_skewnessValue_dec_{i+1}'] = skew(tqwt_coeffs[i] + np.finfo(float).eps)  # Add epsilon for regularization
            features[f'tqwt_kurtosisValue_dec_{i+1}'] = kurtosis(tqwt_coeffs[i] + np.finfo(float).eps)  # Add epsilon for regularization
        else:
            features[f'tqwt_energy_dec_{i+1}'] = 0
            features[f'tqwt_entropy_shannon_dec_{i+1}'] = 0
            features[f'tqwt_entropy_log_dec_{i+1}'] = 0
            features[f'tqwt_TKEO_mean_dec_{i+1}'] = 0
            features[f'tqwt_TKEO_std_dec_{i+1}'] = 0
            features[f'tqwt_medianValue_dec_{i+1}'] = 0
            features[f'tqwt_meanValue_dec_{i+1}'] = 0
            features[f'tqwt_stdValue_dec_{i+1}'] = 0
            features[f'tqwt_minValue_dec_{i+1}'] = 0
            features[f'tqwt_maxValue_dec_{i+1}'] = 0
            features[f'tqwt_skewnessValue_dec_{i+1}'] = 0
            features[f'tqwt_kurtosisValue_dec_{i+1}'] = 0
            
    return features

def compute_features(file_path):
    # Load the audio file
    audio_signal, sample_rate = librosa.load(file_path, sr=None, mono=True)

    # PPE (Pitch Period Entropy)
    ppe = compute_ppe(audio_signal, sample_rate)

    # DFA (Detrended Fluctuation Analysis)
    dfa = nolds.dfa(audio_signal)

    # RPDE (Recurrence Period Density Entropy)
    rpde = 0  # Need to implement this based on specific algorithm

    # numPulses, numPeriodsPulses, meanPeriodPulses, stdDevPeriodPulses
    num_pulses = 0  # Need to implement this based on specific algorithm
    num_periods_pulses = 0  # Need to implement this based on specific algorithm
    mean_period_pulses = 0  # Need to implement this based on specific algorithm
    std_dev_period_pulses = 0  # Need to implement this based on specific algorithm

    # locPctJitter, locAbsJitter, rapJitter, ppq5Jitter, ddpJitter
    loc_pct_jitter = compute_jitter(audio_signal)
    loc_abs_jitter = compute_jitter(np.abs(audio_signal))
    rap_jitter = compute_jitter(np.abs(np.diff(audio_signal)))
    ppq5_jitter = compute_jitter(np.diff(audio_signal)**2)
    ddp_jitter = compute_jitter(np.abs(np.diff(np.abs(audio_signal))))

    # locShimmer, locDbShimmer, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer
    loc_shimmer,loc_db_shimmer,apq3_shimmer,apq5_shimmer,apq11_shimmer,dda_shimmer = compute_shimmer_features(audio_signal, sample_rate)

    # meanAutoCorrHarmonicity, meanNoiseToHarmHarmonicity, meanHarmToNoiseHarmonicity
    harmonic, percussive = librosa.effects.hpss(audio_signal)
    mean_auto_corr_harmonicity = np.mean(librosa.autocorrelate(harmonic))
    mean_noise_to_harm_harmonicity = np.mean(harmonic) / np.mean(percussive)
    mean_harm_to_noise_harmonicity = np.mean(percussive) / np.mean(harmonic)
     
    min_intensity = np.min(audio_signal)
    max_intensity = np.max(audio_signal)
    mean_intensity = np.mean(audio_signal)
    
    f1 = np.mean(audio_signal)  # Mean
    f2 = np.std(audio_signal)   # Standard deviation
    f3 = np.median(audio_signal)  # Median
    f4 = np.percentile(audio_signal, 75)  # 75th percentile
    
    spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate)
    b1 = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate)[0]
    b2 = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate, p=2)[0]
    b3 = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate, p=3)[0]
    b4 = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate, p=4)[0]

    b1 = b1.mean()
    b2 = b2.mean()
    b3 = b3.mean()
    b4 = b4.mean()
    
    data = {
        'PPE': ppe,
        'DFA': dfa,
        'RPDE': rpde,
        'numPulses': num_pulses,
        'numPeriodsPulfses': num_periods_pulses,
        'meanPeriodPulses': mean_period_pulses,
        'stdDevPeriodPulses': std_dev_period_pulses,
        'locPctJitter': loc_pct_jitter,
        'locAbsJitter': loc_abs_jitter,
        'rapJitter': rap_jitter,
        'ppq5Jitter': ppq5_jitter,
        'ddpJitter': ddp_jitter,
        'locShimmer': loc_shimmer,
        'locDbShimmer': loc_db_shimmer,
        'apq3Shimmer': apq3_shimmer,
        'apq5Shimmer': apq5_shimmer,
        'apq11Shimmer': apq11_shimmer,
        'ddaShimmer': dda_shimmer,
        'meanAutoCorrHarmonicity': mean_auto_corr_harmonicity,
        'meanNoiseToHarmHarmonicity': mean_noise_to_harm_harmonicity,
        'meanHarmToNoiseHarmonicity': mean_harm_to_noise_harmonicity,
        'minIntensity': min_intensity,
        'maxIntensity': max_intensity,
        'meanIntensity': mean_intensity,
        'f1':f1,
        'f2':f2,
        'f3':f3,
        'f4':f4,
        'b1':b1,
        'b2':b2,
        'b3':b3,
        'b4':b4
    }
    
    feature_functions = [
        compute_features_vocal_fold,
        compute_audio_features_mfccs,
        calculate_features_wavelet,
        calculate_tqwt_features
    ]

    # Loop through each feature function and merge its result into data
    for function in feature_functions:
        features = function(file_path)
        print(len(features.keys()))
        data = {**data, **features}

    return data


