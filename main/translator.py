"""

"""
# standard library:
# allow user input arguments
import argparse
# use paths 
import pathlib
# check paths and files exist
import os, sys

# custom modules:
import numpy as np 
# the deep learning framework which the models were designed and trained with
import torch
# read and downscale audio files
import librosa
#  save .wav file
import soundfile
# The model class for speech recognition, which can be used to load the pre-trained model
from huggingsound import SpeechRecognitionModel
# convert any audio input to mp4
from moviepy.editor import AudioFileClip
from matplotlib import pyplot as plt


class HiddenPrints:
    """ This class is used to hide print statements (from moviepie.editor.AudioFileClip)
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def check_args(parsed_args):
    """
    """
    # check input mp4 path
    input= parsed_args.Input
    if not os.path.isfile(input):
        raise ValueError("--Input does not exist or was not found", input)
    if input.suffix.upper() not in (".MP4", ".WAV", '.MP3'):
        print(f"\t[WARNING]:--Input does not have a recognized audio/video format. This may cause the script to fail.")

    # check output path
    output = parsed_args.Output
    if os.path.exists(output):
        raise ValueError("--Output already exists", output)
    if output.suffix.upper() != ".SRT":
        print(f"\t[WARNING]:--Output path does not have the .srt format. The output file will be written in the .srt format.")


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

    # do the final split
    if remainder < chunk_length:
        # if the remaining audio is shorter than 1 chunk, save it as a chunk
        last_chunk_data = loaded_audio_time_series[round(seconds_seen*sample_rate) ::]
        soundfile.write(file=f'{save_dir}{chunks_saved}_chunk.wav', data=last_chunk_data, samplerate=sample_rate)
        audio_paths.append(f'{save_dir}{chunks_saved}_chunk.wav')
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
        # last chunk
        chunks_saved += 1
        last_chunk_data = loaded_audio_time_series[round(final_split_second*sample_rate) ::]
        soundfile.write(file=f'{save_dir}{chunks_saved}_chunk.wav', data=last_chunk_data, samplerate=sample_rate)
        audio_paths.append(f'{save_dir}{chunks_saved}_chunk.wav')

    return audio_paths


def transcription_to_SRT_format():
    """
    """ 
    


def create_russian_subtitle_file(model_path, audio_input_path, subtitle_output_path, num_of_cores, certainty_threshold,
                                 chunk_length=10, search_length=4, temporary_save_dir='./main/temp/',
                                 required_sample_khz=16000, temp_reformatted_audio_path='./main/temp/temp_reformatted.wav',
                                 model_name="jonatasgrosman/wav2vec2-large-xlsr-53-russian"):
    """
    """
    # ensure torch doesn't use up too much of the CPU
    torch.set_num_threads(num_of_cores)

    # convert audio to .wav
    print(f"\t[INFO]: Converting input audio to .wav...")
    input_to_wav(source_file=str(audio_input_path.resolve()), target_file=temp_reformatted_audio_path)

    # load audio input
    print(f"\t[INFO]: Loading .wav and converting to sample rate {required_sample_khz} kHz ...")
    loaded_audio_time_series, sample_rate = librosa.load(path=temp_reformatted_audio_path, sr=required_sample_khz)
    duration = librosa.get_duration(y=loaded_audio_time_series, sr=sample_rate)
    
    # split long audio files into parts, so the memory max isnt reached
    print(f"\t[INFO]: Splitting input audio into max {chunk_length}s fragments to prevent memory overload...")
    audio_paths = split_audio_file(save_dir=temporary_save_dir, duration=duration, chunk_length=chunk_length, search_length=search_length, 
                                   loaded_audio_time_series=loaded_audio_time_series, sample_rate=required_sample_khz)

    # load model (will automatically download if it does not exist)
    print(f"\t[INFO]: Loading model...")
    if not os.path.exists(model_path):
        print(f'\t\t[INFO]: The model was not found at the specified path. Downloading and saving...')
        model = SpeechRecognitionModel(model_name)
        # saving the model in the bad (non-state_dict) way because how huggingface models are setup
        torch.save(model, str(model_path.resolve()))
    else:
        # loading the model in the bad (non state_dict) way because how huggingface models are setup
        model = torch.load(model_path)

    print(f"\t[INFO]: Starting transcription...")
    transcriptions = model.transcribe(audio_paths)
    for transcription in transcriptions:
        print(transcription)
        #print(transcription['transcription'])
    
    print(f"\t[INFO]: Converting transcriptions into a subtitle file...")
    print('done')


def run_main(parsed_args):
    """ Run main calls 
    """
    audio_pathlib = parsed_args.Input
    output_path =  parsed_args.Output
    model_path = parsed_args.TrainedModelPath
    thresh = parsed_args.CertaintyThreshold
    num_of_cores = parsed_args.Cores

    create_russian_subtitle_file(model_path=model_path, audio_input_path=audio_pathlib, num_of_cores=num_of_cores,
                                 certainty_threshold=thresh, subtitle_output_path=output_path)


# TODO:  create subtitle function, add autocorrect option, 
# it is probably smart to use long silences (>1s) as 1 sentence.
# add confidence threshold, setup as pip package, add live video processing
# the training data was very short (average length 5 - 10 secs, use this for transcription as well)
#
# https://stackoverflow.com/questions/72575721/how-to-get-letters-position-relative-to-audio-time-in-huggingsound --> how to interpret different timestamps
# https://huggingface.co/blog/asr-chunking --> how to split audio into chunks
if __name__ == "__main__":
    # define the user input
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--Input", required=True, type=pathlib.Path,
                        help='File path to the input .mp4, .mp3, or .wav file to which you want to add Russian subtitles.')
    PARSER.add_argument("--Output", required=False, type=pathlib.Path, default='./output.srt',
                        help='File path to where the output file with subtitles will be created. Default=./output.srt')
    PARSER.add_argument("--TrainedModelPath", required=False, type=pathlib.Path, default='./main/model/pretrained_model.pth',
                        help='File path to where the trained model will be downloaded to and/or loaded from. Default=./model/pretrained_model.pth')
    PARSER.add_argument("--CertaintyThreshold", required=False, type=float, default=0.95,
                            help='???')
    PARSER.add_argument("--Cores", required=False, type=int, default=4, 
                            help='The amount of cores used to translate the audio. Default=6')
    # check the user input
    check_args(PARSER.parse_args())
    # run the main commandline function
    run_main(PARSER.parse_args())
