import os
import sys
import re
from pydub import AudioSegment
from tqdm import tqdm
from utils_vad import *

model_path = sys.argv[1]
wav_dir = "/data/BEA_wavs/mind"

model = init_jit_model(model_path)

# 不符合命名要求的数字集合
excluded_numbers = {"037", "070", "075", "102", "146", "154", "176", "196", "198", "463", 
                    "005", "008", "021", "026", "065", "084", "145", "156", "159", "187", 
                    "190", "202", "206", "212", "296", "387", "001", "006", "014", "031", 
                    "045", "051", "056", "061", "066", "071", "076", "080", "090", "098", 
                    "107", "111", "117", "141", "151", "157", "167", "172", "177", "181", 
                    "186", "193", "200", "205", "462", "002", "007", "023", "032", "047", 
                    "052", "057", "062", "067", "072", "077", "082", "094", "103", "108", 
                    "113", "118", "147", "152", "162", "168", "173", "178", "182", "188", 
                    "194", "201", "207", "464", "003", "009", "024", "036", "049", "053", 
                    "058", "063", "068", "073", "078", "083", "095", "104", "109", "114", 
                    "119", "148", "153", "165", "169", "174", "179", "183", "189", "195", 
                    "203", "208", "004", "010", "029", "038", "050", "055", "060", "064", 
                    "069", "074", "079", "088", "096", "105", "110", "116", "121", "149", 
                    "155", "166", "170", "175", "180", "184", "192", "199", "204", "210"}

# 遍历文件夹中的文件
for filename in os.listdir(wav_dir):
    match = re.search(r'_(\d+)_', filename)
    if match and match.group(1) not in excluded_numbers:
        wav_path_x = os.path.join(wav_dir, filename)
        audio = AudioSegment.from_file(wav_path_x, format="wav")
        wav = read_audio(wav_path_x, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
        
        concatenated_segment = AudioSegment.empty()
        concatenated_duration = 0
        count = 0
        out_dir = "/data/Vad/silero-vad/temp/wav/"

        for item in speech_timestamps:
            start_time = int(item['start'] / 16000 * 1000)
            end_time = int(item['end'] / 16000 * 1000)
            segment = audio[start_time:end_time]
            segment_duration = end_time - start_time

            if concatenated_duration + segment_duration > 30000:
                out_file_idx = filename.split(".")[0] + "_" + str(count).zfill(5) + "_concat"
                concatenated_segment.export(out_dir + out_file_idx + ".wav", format="wav")
                concatenated_segment = segment
                concatenated_duration = segment_duration
                count += 1
            else:
                concatenated_segment += segment
                concatenated_duration += segment_duration

        if concatenated_duration > 0:
            out_file_idx = filename.split(".")[0] + "_" + str(count).zfill(5) + "_concat"
            concatenated_segment.export(out_dir + out_file_idx + ".wav", format="wav")

