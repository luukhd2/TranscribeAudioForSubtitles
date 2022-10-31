""" The main function of this repository.
    Can be run in commandline to specify an input audio file and language.
    
    Example:
    python ./TranscribeAudioForSubtitles/transcribe.py --Input "./TranscribeAudioForSubtitles/input/EN_pale_blue_dot_example.wav" --Language "EN" 
"""
# standard library:
# allow user input arguments
import argparse
# use paths 
import pathlib
# check paths and files exist
import os

# 3rd party modules:
# the deep learning framework which the models were designed and trained with
import torch
# The model class for speech recognition, which can be used to load the pre-trained model
from huggingsound import SpeechRecognitionModel

# import from within this dir
from audio_prep import split_audio_file, input_to_wav, split_audio_file, audio_input_to_wav
from subtitles import get_autocorrected_word_list, get_word_list_from_transcriptions, word_list_to_srt_string, srt_string_to_webvtt
 

def get_hardcoded_language_dict():
    """

    model names found here: https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-english
    """
    return {"EN":"jonatasgrosman/wav2vec2-large-xlsr-53-english",
            "RU":"jonatasgrosman/wav2vec2-large-xlsr-53-russian",
            "PT":"jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
            "FR":"jonatasgrosman/wav2vec2-large-xlsr-53-french",
            "NL":"jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
            "ES":"jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
            "DE":"jonatasgrosman/wav2vec2-large-xlsr-53-german",
            "PL":"jonatasgrosman/wav2vec2-large-xlsr-53-polish",
            "IT":"jonatasgrosman/wav2vec2-large-xlsr-53-italian",
            "JP":"jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
            "FN":"jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
            "GR":"jonatasgrosman/wav2vec2-large-xlsr-53-greek",
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
        if output.suffix.upper() != ".SRT" and output.suffix.upper() != ".VTT":
            print(f"\t[WARNING]:--Output path does not have a supported format (.srt or .vtt)")

    use_autocorrect = parsed_args.Autocorrect
    if use_autocorrect:
        if parsed_args.Language not in ("EN", "PL", "TU", "RU", "UA", "CZ", "PT", "GR", "IT", "FR", "ES", "VN"):
            raise ValueError(f"The specified language, {parsed_args.Language}, is not supported for using autocorrect. Turn the --AutoCorrect feature off")


def create_subtitle_file(model_path, audio_input_path, subtitle_output_path, num_of_cores, word_certainty_threshold,
                        character_certainty_threshold, time_between_subtitles, language, use_autocorrect,
                        separate_voice, subtitle_format,
                        chunk_length=10, search_length=4, temporary_save_dir='./TranscribeAudioForSubtitles/temp/',
                        required_sample_khz=16000, temp_reformatted_audio_path='./TranscribeAudioForSubtitles/temp/temp_reformatted.wav'):
    """
    """
    print(f"[INFO]: Creating subtitles for file {audio_input_path}")

    # ensure torch doesn't use up too much
    torch.set_num_threads(num_of_cores)

    # get the model name
    model_name = get_hardcoded_language_dict()[language]
    if model_path is None:
        model_path = pathlib.Path(f"{temporary_save_dir}{language}_pretrained_model.pth")

    # if needed, make the output file name
    if subtitle_output_path is None:
        subtitle_output_path = f"{str(audio_input_path.resolve()).rpartition('.')[0]}_subtitle{subtitle_format}"

    # convert audio to .wav
    print(f"\t[INFO]: Converting input audio to .wav...")
    input_to_wav(source_file=str(audio_input_path.resolve()), target_file=temp_reformatted_audio_path)

    # load audio input
    print(f"\t[INFO]: Loading .wav and converting to sample rate {required_sample_khz} kHz ...")
    if separate_voice:
        print(f"\t\t[INFO]: Extracting only voices from background noise...")
    loaded_audio_time_series, sample_rate, duration = audio_input_to_wav(temp_reformatted_audio_path=temp_reformatted_audio_path,
                                                                        required_sample_khz=required_sample_khz, use_vocal_separation=separate_voice)

    # split long audio files into parts, so the memory max isnt reached
    print(f"\t[INFO]: Splitting input audio into max {chunk_length}s fragments to prevent memory overload...")
    
    audio_paths, audio_file_durations = split_audio_file(save_dir=temporary_save_dir, duration=duration, chunk_length=chunk_length, search_length=search_length, 
                                   loaded_audio_time_series=loaded_audio_time_series, sample_rate=required_sample_khz)

    # anything to do with the model will use no grad (no weight updates == evaluation)
    with torch.no_grad():
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

        # load model to gpu or cpu, depending on which is available
        # load model on gpu if available
        if torch.cuda.is_available():
            model.device = 'cuda'
            # reload the model so the device is defined everywhere
            model._load_model()

        print(f"\t[INFO]: Starting transcription...")
        transcriptions = model.transcribe(audio_paths)

    print(f"\t[INFO]: Converting transcriptions into a subtitle file...")
    word_list = get_word_list_from_transcriptions(transcriptions, word_certainty_threshold, character_certainty_threshold, audio_file_durations=audio_file_durations)

    if use_autocorrect:
        print(f"\t\t[INFO]: Autocorrecting words for spelling mistakes...")
        word_list = get_autocorrected_word_list(word_list, language)

    srt_string = word_list_to_srt_string(all_transcribed_words=word_list, time_between_subtitles=time_between_subtitles)
    if subtitle_format == '.srt':
        with open(subtitle_output_path, mode='w', encoding='utf-8') as output_file:
            output_file.write(srt_string)
    else:
        vtt_string = srt_string_to_webvtt(srt_string)
        with open(subtitle_output_path, mode='w', encoding='utf-8') as output_file:
            output_file.write(vtt_string)
    print('\t[INFO] done.')


def run_main(parsed_args):
    """ Run main, call the subtitle creation function for either a folder or one specific file.
    Only called when the user uses commandline args to start the function.
    """
    # get arguments
    input_path = parsed_args.Input
    output_path =  parsed_args.Output
    model_path = parsed_args.TrainedModelPath
    language = parsed_args.Language
    character_thresh = parsed_args.CharacterCertaintyThreshold
    word_thresh = parsed_args.WordCertaintyThreshold
    num_of_cores = parsed_args.Cores
    time_between_subtitles = parsed_args.TimeBetweenSubtitles
    use_autocorrect = parsed_args.Autocorrect
    separate_voice = parsed_args.SeparateVoice
    subtitle_format = parsed_args.SubtitleFormat

    # if the path is not a directory, but to a specific file
    if not input_path.is_dir():
        # call the subtitle creation function once
        create_subtitle_file(model_path=model_path, audio_input_path=input_path, num_of_cores=num_of_cores,
                            character_certainty_threshold=character_thresh, word_certainty_threshold=word_thresh, 
                            subtitle_output_path=output_path, time_between_subtitles=time_between_subtitles,
                            language=language, use_autocorrect=use_autocorrect, separate_voice=separate_voice,
                            subtitle_format=subtitle_format)
    # if the path leads to a directory
    else:
        # Iterate directory
        for path in os.listdir(input_path):
            # check if current path is a file
            full_path = os.path.join(input_path, path)
            if os.path.isfile(full_path):
                # check the file meets one of the allowed input formats.
                file_prefix, _ , file_type = full_path.rpartition(".")
                if file_type.upper() in ("MP3", "MP4", "WAV"):
                    create_subtitle_file(model_path=model_path, audio_input_path=pathlib.Path(full_path), num_of_cores=num_of_cores,
                                        character_certainty_threshold=character_thresh, word_certainty_threshold=word_thresh, 
                                        subtitle_output_path=None, time_between_subtitles=time_between_subtitles,
                                        language=language, use_autocorrect=use_autocorrect, separate_voice=separate_voice,
                                        subtitle_format=subtitle_format)
    

# TODO: setup as pip package, add live video processing. Try and find better speech extraction methods.
if __name__ == "__main__":
    # define the user input
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--Input", required=True, type=pathlib.Path,
                        help='File path to the input .mp4, .mp3, or .wav file to which you want to add subtitles.')
    PARSER.add_argument("--Output", required=False, type=pathlib.Path, default=None,
                        help='File path to where the output file with subtitles will be created. Default="AUDIO FILE NAME"_subtitles.')
    PARSER.add_argument("--TrainedModelPath", required=False, type=pathlib.Path, default=None,
                        help='File path to where the trained model will be downloaded to and/or loaded from. Default=./TranscribeAudioForSubtitles/model/"TWO_LETTER_LANGUAGE NAME"_pretrained_model.pth.')
    PARSER.add_argument("--Language", required=False, type=str, default="EN", 
                        help='Language spoken in the audio file. Use the two letter code for the language (default="EN").')
    PARSER.add_argument("--SubtitleFormat", required=False, type=str, default='.vtt',
                           help='Which output subtitle file format should be used. Default=.vtt but .srt is also supported.')
    PARSER.add_argument("--CharacterCertaintyThreshold", required=False, type=float, default=0.5,
                            help='How certain the model must be to include a character prediction (default=0.70), range=0-1.')
    PARSER.add_argument("--WordCertaintyThreshold", required=False, type=float, default=0.5,
                            help='How certain the model must be to include a word prediction (default=0.75), range=0-1.')
    PARSER.add_argument("--Cores", required=False, type=int, default=4, 
                            help='The amount of cores used to translate the audio. Default=4.')
    PARSER.add_argument("--Autocorrect", required=False, type=bool, default=False, 
                            help='Use autocorrect to fix spelling mistakes in transcription. (True or False)')
    PARSER.add_argument("--SeparateVoice", required=False, type=bool, default=False, 
                            help='Process input audio for voice separation (True or False)')
    PARSER.add_argument("--TimeBetweenSubtitles", required=False, type=int, default=800, 
                            help='Time in miliseconds between words that will cause a subtitle split. Decrease if subtitles are too long (default=800).')
    # check the user input
    check_args(PARSER.parse_args())
    # run the main commandline function
    run_main(PARSER.parse_args())
