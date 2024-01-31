# dependencies
# NeuroDSP
from neurodsp.spectral import compute_spectrum #, rotate_powerlaw
# from neurodsp.utils import create_times
# from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.spectral import plot_power_spectra
# from neurodsp.plts.time_series import plot_time_series
# FOOOF
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model
from fooof.utils.reports import methods_report_text
# Other
from scipy.io import loadmat
import numpy as np
# from scipy import signal
from scipy.signal import savgol_filter
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.stats import mode


# load matlab file and append to list
def load_HPC(file_name):
    """
    Load matlab file and append to list
    """
    mat = loadmat(file_name)
    data = mat['HPC']
    return data

def load_PFC(file_name):
    """
    Load matlab file and append to list
    """
    mat = loadmat(file_name)
    data = mat['PFC']
    return data

def load_sleep_score(file_name):
    """
    Load matlab file and append to list
    """
    mat = loadmat(file_name)
    data = mat['states']
    return data

# sampling frequency
fs = 2500

# nomalize data
def normalize(data):
    return (data - np.nanmean(data)) / np.nanstd(data)

# make plot
def make_plot_hpc(rat, day, data, sleep_score):
    """
    makes plot of the smoothed slopes and sleep score
    """
    # plot smoothed slopes day 3
    norm_day_HPC = normalize(data)

    # data_HPC_30_min = data_day2_hpc[:60*60*fs]
    window_length = 3 * fs
    aperiodic_params_HPC = []
    freq_range = [1, 100]

    for start in np.arange(0, len(norm_day_HPC) - window_length + 1, window_length):
        window = norm_day_HPC[start:start + window_length]
        window = np.ravel(window)
        freq_mean, psd_mean = compute_spectrum(window, fs, method='welch', avg_type='mean', nperseg=window_length)
        fm = FOOOF()
        fm.fit(freq_mean, psd_mean, freq_range)
        aperiodic_params_HPC.append(fm.get_params('aperiodic_params'))

    # get the slope of the aperiodic component
    slopes = [param[0] for param in aperiodic_params_HPC]

    # nomralize the slope
    slopes_norm = zscore(slopes)
    # smooth the slope
    slopes_smooth = savgol_filter(slopes_norm, window_length=101, polyorder=5)

    # Score labels and their numerical mapping
    score_labels = {1: 'Wake', 3: 'NREM', 4: 'Intermediate', 5: 'REM'}
    num_labels = {label: num for num, label in enumerate(score_labels.values(), start=1)}

    # Remove last element to ensure sleep_score can be reshaped
    rm_elem = len(sleep_score[0]) % 3
    if rm_elem > 0:
        reshaped_scores = sleep_score[0][:-rm_elem].reshape(-1, 3)
    else:
    
        reshaped_scores = sleep_score.reshape(-1, 3)
    
    # Apply majority voting
    majority_scores = mode(reshaped_scores, axis=1).mode.flatten()

    # Ensure all scores in majority_scores are in score_labels
    assert all(score in score_labels for score in majority_scores), "Some scores are not in score_labels"

    # Map categorical labels to numerical values for plotting
    mapped_scores = np.array([num_labels[score_labels[score]] for score in majority_scores])

    # Time arrays for x-axis in minutes
    time_minutes = np.arange(0, len(slopes_smooth)) * 3 / 60  # Convert to minutes
    sleep_time_minutes = np.arange(0, len(majority_scores)) * 3 / 60  # Convert to minutes

    plt.figure(figsize=(20, 10))

    # First subplot - Smoothed Slopes
    plt.subplot(2, 1, 1)
    plt.plot(time_minutes, slopes_smooth, marker='o', linestyle='-', color='black', label='Smoothed Slopes')
    plt.title("Smoothed Slope of Aperiodic Signal Component Over Time (HPC) | 1-100 HZ | Rat" + str(rat)+ " Day" + str(day))
    plt.xlabel("Time (minutes)")
    plt.ylabel("Smoothed Slope")
    plt.grid(True)
    plt.legend()

    # Second subplot - Sleep Scoring
    plt.subplot(2, 1, 2)
    plt.step(sleep_time_minutes, mapped_scores, where='mid')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Sleep Stage')
    plt.title('Majority Voting Sleep Stages Over Time | Rat' + str(rat) + 'Day' + str(day))
    plt.yticks(list(num_labels.values()), list(score_labels.values()))  # Setting y-ticks to stage labels
    plt.xticks(np.arange(0, max(sleep_time_minutes) + 1, 5))  # Marking every 5 minutes
    plt.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    # save the plot
    plt.savefig('data/plots/' + str(rat) +'_'+ str(day) +'_PFC_100_CH22_0.continuous.png')

def slopes(data):
    """
    Gets the slopes of the aperiodic component of the power spectrum
    """
    norm_day_HPC = normalize(data)

    # data_HPC_30_min = data_day2_hpc[:60*60*fs]
    window_length = 3 * fs
    aperiodic_params_HPC = []
    freq_range = [1, 100]

    for start in np.arange(0, len(norm_day_HPC) - window_length + 1, window_length):
        window = norm_day_HPC[start:start + window_length]
        window = np.ravel(window)
        freq_mean, psd_mean = compute_spectrum(window, fs, method='welch', avg_type='mean', nperseg=window_length)
        fm = FOOOF()
        fm.fit(freq_mean, psd_mean, freq_range)
        aperiodic_params_HPC.append(fm.get_params('aperiodic_params'))

    # get the slope of the aperiodic component
    slopes = [param[0] for param in aperiodic_params_HPC]

    # nomralize the slope
    slopes_norm = zscore(slopes)
    # smooth the slope
    slopes_smooth = savgol_filter(slopes_norm, window_length=101, polyorder=5)
    return slopes_norm, slopes_smooth

def majority_scoring(sleep_score):
    """
    Apply majority voting to sleep score
    """
    # Remove last element to ensure sleep_score can be reshaped
    rm_elem = len(sleep_score[0]) % 3
    if rm_elem > 0:
        reshaped_scores = sleep_score[0][:-rm_elem].reshape(-1, 3)
    else:
    
        reshaped_scores = sleep_score.reshape(-1, 3)
    
    # Apply majority voting
    majority_scores = mode(reshaped_scores, axis=1).mode.flatten()
    return majority_scores


def plot_average(rat, day, sleep_score, data, type):
    """
    Plot average of smoothed  and unsmoothed slopes for each sleep stage in an errorbar plot
    """
    # Get smoothed and unsmoothed slopes
    data_smoothed = slopes(data)[1]
    data_unsmoothed = slopes(data)[0]

    # Convert to numpy arrays
    smooth = np.asarray(data_smoothed)
    unsmooth = np.asarray(data_unsmoothed)
    majority_scores = majority_scoring(sleep_score)

    # Prepare data for plotting
    means_smoothed = []
    stds_smoothed = []
    means_unsmoothed = []
    stds_unsmoothed = []

    score_labels = {1: 'Wake', 3: 'NREM', 4: 'Intermediate', 5: 'REM'}
    for score, label in score_labels.items():
        # Select relevant slopes for the current sleep stage
        relevant_indices = majority_scores == score

        # relevant indices len = 3600
        # ajust lenght of smooth and unsmooth to 3600
        smooth = smooth[:3600]
        unsmooth = unsmooth[:3600]

        # Smoothed data
        relevant_slopes_smoothed = smooth[relevant_indices]
        means_smoothed.append(np.nanmean(relevant_slopes_smoothed))
        stds_smoothed.append(np.nanstd(relevant_slopes_smoothed))

        # Unsmoothed data
        relevant_slopes_unsmoothed = unsmooth[relevant_indices]
        means_unsmoothed.append(np.nanmean(relevant_slopes_unsmoothed))
        stds_unsmoothed.append(np.nanstd(relevant_slopes_unsmoothed))

    # Plotting for Smoothed Data
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.errorbar(range(1, len(score_labels)+1), means_smoothed, yerr=stds_smoothed, fmt='-ok', markersize=10,
                markeredgecolor='black', markerfacecolor='black', label='mean of slopes')
    plt.ylabel("Slope, z")
    plt.title("Smoothed")
    plt.grid(True)
    plt.xticks(ticks=range(1, len(score_labels)+1), labels=score_labels.values())
    plt.xlim(0, len(score_labels) + 1)  # Adjust based on your data range
    plt.legend()

    # Plotting for Unsmoothed Data
    plt.subplot(1, 2, 2)
    plt.errorbar(range(1, len(score_labels)+1), means_unsmoothed, yerr=stds_unsmoothed, fmt='-ok', markersize=10,
                markeredgecolor='black', markerfacecolor='blue', linestyle='-', label='mean of slopes')
    plt.ylabel("Slope, z")
    plt.title("Unsmoothed")
    plt.grid(True)
    plt.xticks(ticks=range(1, len(score_labels)+1), labels=score_labels.values())
    plt.xlim(0, len(score_labels) + 1)  # Adjust based on your data range
    plt.legend()
    
    plt.suptitle("Average Slopes for Each Sleep Stage (" + str(type)+ ") | Rat " + str(rat) + " Day " + str(day))
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig('data/plots/' + str(rat) +'_'+ str(day) +'_'+ str(type) +'_average.png')
  

# rat 3 

# days = [2, 3, 4, 5, 8, 17]
# trials = [load_HPC('data/rat3/day2/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat3/day3/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat3/day4/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat3/day5/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat3/day8/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat3/day17/HPC_100_CH18_0.continuous.mat')]
# sleep_score = [load_sleep_score('data/rat3/day2/2021-06-03_13-34-04_posttrial5-states_ES2.mat'),
#                load_sleep_score('data/rat3/day3/2021-06-04_13-43-50_posttrial5-states_ES2.mat'),
#                load_sleep_score('data/rat3/day4/2021-06-07_14-23-57_posttrial5-states_ES.mat'),
#                load_sleep_score('data/rat3/day5/2021-06-09_13-36-38_posttrial5-states_ES.mat'),
#                load_sleep_score('data/rat3/day8/2021-06-16_11-46-46_posttrial5-states_ES.mat'),
#                load_sleep_score('data/rat3/day17/2021-07-04_15-07-11_posttrial5-states_ES.mat')]
# rat3 = list(zip(days, trials, sleep_score))

# rat 4 (done HPC and PFC)
# days = [2, 5, 6, 9, 16, 17]
# trials = [load_HPC('data/rat4/day2/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat4/day5/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat4/day6/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat4/day9/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat4/day16/HPC_100_CH18_0.continuous.mat'),
#           load_HPC('data/rat4/day17/HPC_100_CH18_0.continuous.mat')]
# sleep_score = [load_sleep_score('data/rat4/day2/2021-06-03_13-38-14_posttrial5-states_SM.mat'),
#                load_sleep_score('data/rat4/day5/2021-06-09_13-47-17_posttrial5-states_SM.mat'),
#                load_sleep_score('data/rat4/day6/2021-06-10_14-03-03_posttrial5-states_SM.mat'),
#                load_sleep_score('data/rat4/day9/2021-06-18_14-00-33_posttrial5-states_SM.mat'),
#                load_sleep_score('data/rat4/day16/2021-07-02_15-34-05_posttrial5-states_SM.mat'),
#                load_sleep_score('data/rat4/day17/2021-07-04_15-23-42_posttrial5-states_SM.mat')]
# rat4 = list(zip(days, trials, sleep_score))

# rat 5 (done hpc and pfc)
# days = [8, 14, 15]
# trials = [load_HPC('data/rat5/day8/HPC_100_CH14_0.continuous.mat'),
#           load_HPC('data/rat5/day14/HPC_100_CH14_0.continuous.mat'),
#           load_HPC('data/rat5/day15/HPC_100_CH14_0.continuous.mat')]
# sleep_score = [load_sleep_score('data/rat5/day8/2021-07-21_15-14-50_posttrial5-states_ES.mat'),
#                load_sleep_score('data/rat5/day14/2021-07-29_16-00-16_posttrial5-states_ES.mat'),
#                load_sleep_score('data/rat5/day15/2021-08-01_15-51-31_posttrial5-states_ES.mat')]
# rat5 = list(zip(days, trials, sleep_score))


# run for every rat:
    
# for day, data, sleep_score in rat3:
#     rat = 3
#     type = 'HPC'
#     plot_average(rat, day, sleep_score, data, type)
#     print('rat' + str(rat) + ' day' + str(day) + ' done')


# day 2 rat 5
# error message: fooof.core.errors.DataError: The input power spectra data, after logging, contains NaNs or Infs. 
# This will cause the fitting to fail. One reason this can happen is if inputs are already logged. 
# Inputs data should be in linear spacing, not log.

# data = load_HPC('data/rat5/day2/HPC_100_CH14_0.continuous.mat')
# sleep_score = load_sleep_score('data/rat5/day2/2021-07-09_16-28-31_posttrial5-states_ES.mat'),
# day = 2
   

# day 3 and 4 rat 5
# error message: AssertionError: Some scores are not in score_labels
# load_HPC('data/rat5/day3/HPC_100_CH14_0.continuous.mat'),
# load_HPC('data/rat5/day4/HPC_100_CH14_0.continuous.mat'),
# load_sleep_score('data/rat5/day3/2021-07-14_16-17-11_posttrial5-states_ES.mat'),
# load_sleep_score('data/rat5/day4/2021-07-16_14-49-56_posttrial5-states_ES.mat'),
            