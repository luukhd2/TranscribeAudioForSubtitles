""" Every custom function used to generate subtitles files from the model predictions.
When ran as main, can be used to convert .srt format to .vtt format.
"""
import copy
import argparse
import pathlib

# use autocorrect to fix spelling mistakes
from spellchecker import SpellChecker


def get_word_list_from_transcriptions(transcriptions, word_thresh, char_thresh, audio_file_durations):
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
    
    # for each transcription list (transcribed audio chunk)
    for transcription_index, transcription in enumerate(transcriptions):
        # count the time before the current fragment
        if transcription_index > 0:
            previous_milliseconds += audio_file_durations[transcription_index-1]*1000
        else:
            previous_milliseconds = 0

        # if the transcription has no predicted words
        if transcription['transcription'] is None or transcription['start_timestamps'] is None \
        or transcription['end_timestamps'] is None or transcription['probabilities'] is None:
            continue
            
        # get transcription info
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
            if prob >= char_thresh or character == " " and prob > char_thresh / 2:
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


def get_autocorrected_word_list(word_list, language):
    """
    """       
    #calculating the hamming distances and similarities for each word of the sentence
    #with each of the chosen keywords contained in list
    
    checker = SpellChecker(language=language.lower(), distance=3)
    new_word_list = copy.deepcopy(word_list)
    for word in new_word_list:
        correction = checker.correction(word['word'])
        if correction is not None:
            word['word'] = correction
    
    return new_word_list


def srt_string_to_webvtt(subtitles):
    """
    The main use of this function is reformatting .srt files to .vtt files,
    because that is one of the only formats Chromecast supports...

    subtitles: <str> the full contents of a .srt subtitle file.
    returns: <str> the full contents of a .vtt (Web Video Text Tracks Format) subtitle file.
    """

    output = f"WEBVTT\n\n"

    # get all subtitles sections
    sections = [section for section in subtitles.split("\n\n") if section]
    # go over each section and reformat
    for section in sections:
        # skip sections that somehow has an incorrect format and does not have 3 lines.
        section_lines = section.split("\n")
        if len(section_lines) != 3:
            continue
        section_number = section_lines[0]
        output = f"{output}{section_number}\n"
        
        time_string = section_lines[1]
        time_string = time_string.replace(",", ".")
        output = f"{output}{time_string}\n"

        sentence = section_lines[2]
        sentence = f"- {sentence}"
        output = f"{output}{sentence}\n\n"
    
    return output


def convert_srt_file_to_vtt_file(srt_path, output_vtt):
    """
    """
    with open(srt_path, 'r', encoding='utf-8') as srt_file:
        srt_info = srt_file.read()

    webvtt_info = srt_string_to_webvtt(srt_info)

    with open(output_vtt, 'w', encoding='utf-8') as output_vtt_file:
        output_vtt_file.write(webvtt_info)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--Input_srt", required=True, type=pathlib.Path,
                        help='File path to the .srt file which you want to convert to .vtt')
    PARSER.add_argument("--Output_vtt", required=True, type=pathlib.Path,
                        help='File path to where the reformatted .vtt subtitle file will be created.')
    PARSED_ARGS = PARSER.parse_args()
    convert_srt_file_to_vtt_file(srt_path=PARSED_ARGS.Input_srt, output_vtt=PARSED_ARGS.Output_vtt)
