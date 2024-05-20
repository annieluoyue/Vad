import os
from utils_vad import *
import sys
import re
from pydub import AudioSegment
from tqdm import tqdm

model_path = sys.argv[1]
wav_path = sys.argv[2]

model = init_jit_model(model_path)
out_wav_file = open(wav_path + ".cut", 'w', encoding="utf-8")

with open(wav_path, 'r', encoding="utf-8") as file:
    for line in tqdm(file.readlines()):
        line = line.strip()
        idx = line.split(" ")[0]
        wav_path_x = line.split(" ")[1]

        audio = AudioSegment.from_file(wav_path_x, format="wav")
        wav = read_audio(wav_path_x, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
        
        concatenated_segment = AudioSegment.empty()
        concatenated_duration = 0
        count = 0
        out_dir = "/data/Vad/silero-vad/temp/wav/"

        for item in speech_timestamps:
            start_time = int(item['start'] / 16000 * 1000)  # 切割起始时间（单位：毫秒）
            end_time = int(item['end'] / 16000 * 1000)  # 切割结束时间（单位：毫秒）
            segment = audio[start_time:end_time]
            segment_duration = end_time - start_time

            # 累积时长接近30秒时输出
            if concatenated_duration + segment_duration > 30000:  # 如果累积长度超过30秒
                out_file_idx = idx + "_" + str(count).zfill(5) + "_concat"
                concatenated_segment.export(out_dir + out_file_idx + ".wav", format="wav")
                out_wav_file.write(out_file_idx + " " + out_dir + out_file_idx + ".wav" + "\n")
                concatenated_segment = segment  # 从当前段开始新的连接
                concatenated_duration = segment_duration
                count += 1
            else:
                concatenated_segment += segment  # 连接段
                concatenated_duration += segment_duration

        # 处理最后一段音频
        if concatenated_duration > 0:
            out_file_idx = idx + "_" + str(count).zfill(5) + "_concat"
            concatenated_segment.export(out_dir + out_file_idx + ".wav", format="wav")
            out_wav_file.write(out_file_idx + " " + out_dir + out_file_idx + ".wav" + "\n")

out_wav_file.close()

