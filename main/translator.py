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


def get_hardcoded_language_dict():
    """

    model names found here: https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-english
    """
    return {"EN":"jonatasgrosman/wav2vec2-xls-r-1b-english",
            "RU":"jonatasgrosman/wav2vec2-large-xlsr-53-russian",
            "PT":"jonatasgrosman/wav2vec2-xls-r-1b-portuguese",
            "FR":"jonatasgrosman/wav2vec2-xls-r-1b-french",
            "NL":"jonatasgrosman/wav2vec2-xls-r-1b-dutch",
            "ES":"jonatasgrosman/wav2vec2-xls-r-1b-spanish",
            "DE":"jonatasgrosman/wav2vec2-xls-r-1b-german",
            "PL":"jonatasgrosman/wav2vec2-xls-r-1b-polish",
            "IT":"jonatasgrosman/wav2vec2-xls-r-1b-italian",
            }


def check_args(parsed_args):
    """
    """
    # check input mp4 path
    input= parsed_args.Input
    if input.is_dir():
        print("[WARNING]: Specified path is a folder, will create output files for each folder")
    else:
        if not os.path.isfile(input):
                raise ValueError("--Input does not exist or was not found", input)
        if input.suffix.upper() not in (".MP4", ".WAV", '.MP3'):
            print(f"\t[WARNING]:--Input does not have a recognized audio/video format. This may cause the script to fail.")

    # check output path
    output = parsed_args.Output
    if output is not None:
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


def get_word_list_from_transcriptions(transcriptions, word_thresh, char_thresh):
    """

    transcriptions: <list of dicts> transcriptions belonging to 1 audio file

    """
    def nested_save_word(word_dict, word_thresh, all_words):
        """ Nested function (only accesable within this function)
        Store the word if is not empty and it meets the word threshold requirements.
        """
        # if the word has letters
        if len(word_dict['word']) > 0:
            # if the word confidence meets the word confidence threshold
            if sum(word_dict['confidences']) / len(word_dict['confidences']) > word_thresh:
                # store the word
                all_words.append(word_dict)

    def nested_get_reset_word_dict():
        """ Get a dictionary in which the info for 1 transcribed word will be stored.
        """
        return  {'word':"", 'start':None, 'end':None, 'confidences':list()}


    all_words = list()
    previous_milliseconds = 0
    # for each transcription list (transcribed audio chunk)
    for transcription in transcriptions:
        sentence = transcription['transcription']
        starts = transcription['start_timestamps']
        ends = transcription['end_timestamps']
        probs = transcription['probabilities']
        
        # initiate the empty word dict (info for 1 word)
        word_dict = nested_get_reset_word_dict()

        # for each predicted character (with any confidence level)    
        for character, start, end, prob in zip(sentence, starts, ends, probs):
            # if the word has no start yet, give it the current start
            if word_dict['start'] is None:
                # keep track of the milliseconds over the full audio clip (including previous chunks)
                word_dict['start'] = previous_milliseconds + start

            # if the predicted character was predicted with enough confidence
            if prob >= char_thresh:
                # if the character was a space (word separator)
                if character == " ":
                    # store the word
                    word_dict['end'] = previous_milliseconds + end
                    nested_save_word(word_dict=word_dict, word_thresh=word_thresh, all_words=all_words)
                    # reset the word
                    word_dict = nested_get_reset_word_dict()
                # extend the current word
                else:
                    word_dict['confidences'].append(prob)
                    word_dict['word'] = f'{word_dict["word"]}{character}'
        
        # store the last word
        word_dict['end'] = previous_milliseconds + end
        nested_save_word(word_dict=word_dict, word_thresh=word_thresh, all_words=all_words)

        # keep track of the milliseconds over the full audio clip (including previous chunks)
        previous_milliseconds += end

    return all_words


def word_list_to_srt_string(all_transcribed_words, time_between_subtitles):
    """

    time_between_subtitles: <float> seconds between subtitles
    used this definition of SRT file Structure: https://docs.fileformat.com/video/srt/
    """
    def nested_milliseconds_to_srt_time(milliseconds):
        """ millisecs to string in format hours:minutes:seconds,milliseconds
        """
        millis = int(milliseconds)
        millis_str = str(millis%1000).zfill(4)[1:4]
        seconds=(millis/1000)%60
        seconds = str(int(seconds)).zfill(2)
        minutes=(millis/(1000*60))%60
        minutes = str(int(minutes)).zfill(2)
        hours=(millis/(1000*60*60))%24
        hours=str(int(hours)).zfill(2)

        # format example: "00:05:15,300" = 5 minutes, 15 seconds, 300 milliseconds
        return f"{hours}:{minutes}:{seconds},{millis_str}"


    def nested_get_SRT_subtitle_string(subtitle_position, subtitle_words):
        """ Transform the subtitle position and the list of subtitle words into a subtitle string with
        the appropriate SRT format.
        """
        # 1) start with the position (an integer)
        subtitle_string = f"{subtitle_position}\n"

        # 2) calculate the subtitle appearance time (00:05:00,400 --> 00:05:15,300)
        first_word = subtitle_words[0]
        start_time = first_word['start']
        start_time_str = nested_milliseconds_to_srt_time(start_time)
        
        last_word = subtitle_words[-1]
        last_time = last_word['end']
        last_time_str = nested_milliseconds_to_srt_time(last_time)

        subtitle_string = f"{subtitle_string}{start_time_str} --> {last_time_str}\n"

        # 3) add the actual words to the suttile
        words_str = " ".join(word['word'] for word in subtitle_words)
        words_str = words_str.capitalize()
        subtitle_string = f"{subtitle_string}{words_str}\n"

        # 4) add another blank line to indicate the end of the subtitle
        subtitle_string = f"{subtitle_string}\n"
        return subtitle_string


    # check input
    if len(all_transcribed_words) < 0:
        print("\t[WARNING]: No words were predicted for this audio file. SRT file will be empty")
        return ""
    
    # define vars
    output = ""
    subtitle_position=1
    subtitle_words = list()

    # for each predicted word
    for word_index, word in enumerate(all_transcribed_words):
        # skip the first word
        if word_index < 1:
            subtitle_words.append(word)
            continue
        # get the previous word
        else:
            previous_word = all_transcribed_words[word_index-1]
        
        # if the time between words is long enough to create a new subtitle
        if abs(word['end'] - previous_word['end']) > time_between_subtitles:
            # create a new subtitle
            subtitle_string = nested_get_SRT_subtitle_string(subtitle_position, subtitle_words)
            # add the subtitle string to the full file
            output = f"{output}{subtitle_string}"

            # prepare variables for next subtitle
            subtitle_position+=1
            subtitle_words = list()
            subtitle_words.append(word)
        # else if the word is still close enough to the active subtitle / sentence
        else:
            subtitle_words.append(word)

    # don't forget the last subtitle
    if len(subtitle_words) > 0:
        # create a new subtitle
        subtitle_string = nested_get_SRT_subtitle_string(subtitle_position, subtitle_words)
        # add the subtitle string to the full file
        output = f"{output}{subtitle_string}"

    return output


def create_subtitle_file(model_path, audio_input_path, subtitle_output_path, num_of_cores, word_certainty_threshold,
                        character_certainty_threshold, time_between_subtitles, language,
                        chunk_length=10, search_length=4, temporary_save_dir='./main/temp/',
                        required_sample_khz=16000, temp_reformatted_audio_path='./main/temp/temp_reformatted.wav'):
    """
    """
    print(f"[INFO]: Creating subtitles for file {audio_input_path}")

    # ensure torch doesn't use up too much of the CPU
    torch.set_num_threads(num_of_cores)

    # get the model name
    model_name = get_hardcoded_language_dict()[language]
    if model_path is None:
        model_path = pathlib.Path(f"./main/model/{language}_pretrained_model.pth")

    # if needed, make the output file name
    if subtitle_output_path is None:
        subtitle_output_path = f"{str(audio_input_path.resolve()).rpartition('.')[0]}_subtitle.srt"

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

    print(f"\t[INFO]: Converting transcriptions into a subtitle file...")
    word_list = get_word_list_from_transcriptions(transcriptions, word_certainty_threshold, character_certainty_threshold)
    srt_string = word_list_to_srt_string(all_transcribed_words=word_list, time_between_subtitles=time_between_subtitles)
    
    with open(subtitle_output_path, mode='w', encoding='utf-8') as output_file:
        output_file.write(srt_string)
    print('\t[INFO]')


def run_main(parsed_args):
    """ Run main calls 
    """
    input_path = parsed_args.Input
    output_path =  parsed_args.Output
    model_path = parsed_args.TrainedModelPath
    language = parsed_args.Language
    character_thresh = parsed_args.CharacterCertaintyThreshold
    word_thresh = parsed_args.WordCertaintyThreshold
    num_of_cores = parsed_args.Cores
    time_between_subtitles = parsed_args.TimeBetweenSubtitles

    if not input_path.is_dir():    
        create_subtitle_file(model_path=model_path, audio_input_path=input_path, num_of_cores=num_of_cores,
                            character_certainty_threshold=character_thresh, word_certainty_threshold=word_thresh, 
                            subtitle_output_path=output_path, time_between_subtitles=time_between_subtitles,
                            language=language)
    else:
        # Iterate directory
        for path in os.listdir(input_path):
            # check if current path is a file
            full_path = os.path.join(input_path, path)
            if os.path.isfile(full_path):
                file_prefix, _ , file_type = full_path.rpartition(".")
                if file_type.upper() in ("MP3", "MP4", "WAV", "MOV", "AVI", "WMV"):
                    create_subtitle_file(model_path=model_path, audio_input_path=pathlib.Path(full_path), num_of_cores=num_of_cores,
                                        character_certainty_threshold=character_thresh, word_certainty_threshold=word_thresh, 
                                        subtitle_output_path=None, time_between_subtitles=time_between_subtitles,
                                        language=language)
        

# TODO:  add autocorrect option, 
# add folder predict option
# setup as pip package, add live video processing
# the training data was very short (average length 5 - 10 secs, use this for transcription as well)
#
# https://stackoverflow.com/questions/72575721/how-to-get-letters-position-relative-to-audio-time-in-huggingsound --> how to interpret different timestamps
# https://huggingface.co/blog/asr-chunking --> how to split audio into chunks
if __name__ == "__main__":
    # define the user input
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--Input", required=True, type=pathlib.Path,
                        help='File path to the input .mp4, .mp3, or .wav file to which you want to add Russian subtitles.')
    PARSER.add_argument("--Output", required=False, type=pathlib.Path, default=None,
                        help='File path to where the output file with subtitles will be created. Default="AUDIO FILE NAME"_subtitles.srt')
    PARSER.add_argument("--TrainedModelPath", required=False, type=pathlib.Path, default=None,
                        help='File path to where the trained model will be downloaded to and/or loaded from. Default=./main/model/"TWO_LETTER_LANGUAGE NAME"_pretrained_model.pth')
    PARSER.add_argument("--Language", required=False, type=str, default="EN", 
                        help='Language spoken in the audio file.')
    PARSER.add_argument("--CharacterCertaintyThreshold", required=False, type=float, default=0.5,
                            help='???')
    PARSER.add_argument("--WordCertaintyThreshold", required=False, type=float, default=0.5,
                            help='???')
    PARSER.add_argument("--Cores", required=False, type=int, default=4, 
                            help='The amount of cores used to translate the audio. Default=4')
    PARSER.add_argument("--TimeBetweenSubtitles", required=False, type=int, default=1000, 
                            help='miliseconds')
    # check the user input
    check_args(PARSER.parse_args())
    # run the main commandline function
    run_main(PARSER.parse_args())
