""" All custom functions involved with preparing audio input for transcript prediction.
"""
import sys
import soundfile
from moviepy.editor import AudioFileClip
import numpy as np
import librosa
import os


class HiddenPrints:
    """ This class is used to hide print statements (from moviepie.editor.AudioFileClip)
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def input_to_wav(source_file, target_file):
    with HiddenPrints():
        audioclip = AudioFileClip(source_file)
        audioclip.write_audiofile(target_file)


def convert_librosa_audio_time_series_to_decibels(audio_time_series):
    """
    audio_time_series: the .wav timeseries loaded with librosa.load()
    returns: numpy array of equal length with the decibel estimate
    """
    def nested_to_decibels(value):
        ref = 1
        if value!=0:
            return 20 * np.log10(abs(value) / ref)
        else:
            return -60

    return np.array(list(map(nested_to_decibels, audio_time_series)))


def split_audio_file(save_dir, duration, chunk_length, search_length, loaded_audio_time_series, sample_rate):
    """

    chunk_length: <float> The max amount of seconds a (split) audio chunk will be.
    search_length: <int> The width (seconds) of the window where the script will find
                         the most quiet point to split the audio on.
    frame_length: <int> the frame length used to estimate audio loudness
    """
    seconds_seen = 0
    remainder = duration
    chunks_saved = 0
    # store the audio chunk files made in a list to retain proper order
    audio_paths = list()
    audio_file_durations = list()

    # estimate 'loudness' of the audio
    decibels = convert_librosa_audio_time_series_to_decibels(audio_time_series=loaded_audio_time_series)
    
    # while we are not at the last split site
    while remainder > (2*chunk_length):
        # get the audio fragment in which we will split on the most quiet point
        search_window_start_index = round((seconds_seen+chunk_length-search_length)*sample_rate)
        search_window_end_index = round((seconds_seen+chunk_length)*sample_rate)
        
        # within the split window, find at which index the most quiet point is
        decibel_split_window = decibels[search_window_start_index : search_window_end_index]
        time_series_split_index = np.argmin(decibel_split_window)
        
        # calculate the duration (seconds) of the current chunk
        chunk_duration = chunk_length-search_length + time_series_split_index/sample_rate

        # get and save the audio time series for this chunk
        trimmed_data = loaded_audio_time_series[round(seconds_seen*sample_rate) : round((seconds_seen+chunk_duration)*sample_rate)]
        soundfile.write(file=f'{save_dir}{chunks_saved}_chunk.wav', data=trimmed_data, samplerate=sample_rate)
        audio_paths.append(f'{save_dir}{chunks_saved}_chunk.wav')
        chunks_saved += 1

        # calculate the overall time values
        seconds_seen = seconds_seen + chunk_duration 
        remainder = duration - seconds_seen
        # store the duration of this chunk
        audio_file_durations.append(chunk_duration)

    # do the final split
    if remainder < chunk_length:
        # if the remaining audio is shorter than 1 chunk, save it as a chunk
        last_chunk_data = loaded_audio_time_series[round(seconds_seen*sample_rate) ::]
        soundfile.write(file=f'{save_dir}{chunks_saved}_chunk.wav', data=last_chunk_data, samplerate=sample_rate)
        audio_paths.append(f'{save_dir}{chunks_saved}_chunk.wav')
        # store the duration of this chunk
        audio_file_durations.append(remainder)

    else:
        # if the remaining audio is longer than 1 chunk, look for a split point in the centre of the remaining audio
        start_second_of_final_split_window = seconds_seen + remainder/2 - search_length/2
        end_second_of_final_split_window = seconds_seen + remainder/2 + search_length/2
        final_decibel_split_window = decibels[round(start_second_of_final_split_window*sample_rate) : round(end_second_of_final_split_window*sample_rate)]
        #                    '                     start               ' + '         seconds inside the split window          ' 
        final_split_second = seconds_seen + remainder/2 - search_length/2 + np.argmin(final_decibel_split_window) / sample_rate 
        # first to last chunk
        first_to_last_chunk_data = loaded_audio_time_series[round(seconds_seen*sample_rate) : int(final_split_second*sample_rate)]
        soundfile.write(file=f'{save_dir}{chunks_saved}_chunk.wav', data=first_to_last_chunk_data, samplerate=sample_rate)
        audio_paths.append(f'{save_dir}{chunks_saved}_chunk.wav')
        first_to_last_chunk_length = final_split_second-seconds_seen
        audio_file_durations.append(first_to_last_chunk_length)
        seconds_seen = seconds_seen + first_to_last_chunk_length

        # last chunk
        chunks_saved += 1
        last_chunk_data = loaded_audio_time_series[round(final_split_second*sample_rate) ::]
        soundfile.write(file=f'{save_dir}{chunks_saved}_chunk.wav', data=last_chunk_data, samplerate=sample_rate)
        audio_paths.append(f'{save_dir}{chunks_saved}_chunk.wav')
        audio_file_durations.append(duration-seconds_seen)
    
    return audio_paths, audio_file_durations


def audio_input_to_wav(temp_reformatted_audio_path, required_sample_khz, use_vocal_separation):
    """
    
    Voice extraction taken from:
    https://stackoverflow.com/questions/49279425/extract-human-vocals-from-song
    """
    # load audio through librosa
    loaded_audio_time_series, sample_rate = librosa.load(path=temp_reformatted_audio_path, sr=required_sample_khz)
    duration = librosa.get_duration(y=loaded_audio_time_series, sr=sample_rate)
    # return the loaded audio
    if not use_vocal_separation:
        return loaded_audio_time_series, sample_rate, duration

    # except if the voices must be filtered out from all audio
    else:
        S_full, phase = librosa.magphase(librosa.stft(loaded_audio_time_series))
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(3, sr=sample_rate)))
        S_filter = np.minimum(S_full, S_filter)
        margin_v = 4 # 4 was 10, made less extreme
        power = 1 # was 2, but turned audio too 'robotic' which worsened performance

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)
        # Once we have the masks, simply multiply them with the input spectrum
        # to separate the components
        S_foreground = mask_v * S_full

        new_y = librosa.istft(S_foreground*phase)
        return new_y, sample_rate, duration
