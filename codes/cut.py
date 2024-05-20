import os
from utils_vad import *
import sys
import re
from pydub import AudioSegment
from tqdm import tqdm

model_path = sys.argv[1]
wav_path = sys.argv[2]

model = init_jit_model(model_path)
out_wav_file = open(wav_path+".cut",'w',encoding="utf-8")

with open(wav_path,'r',encoding="utf-8") as file:

    for line in tqdm(file.readlines()):
        line = line.strip()
        idx = line.split(" ")[0]
        wav_path_x = line.split(" ")[1]

        audio = AudioSegment.from_file(wav_path_x, format="wav")

        wav = read_audio(wav_path_x, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
        print(speech_timestamps)
            
        count = 0
        for item in speech_timestamps:

            out_dir = "/data/Vad/silero-vad/temp/wav/"
            start_time = int(item['start'] / 16000 * 1000)  # 切割起始时间（单位：毫秒）
            end_time = int(item['end'] /16000 * 1000)  # 切割结束时间（单位：毫秒）
            
            segment = audio[start_time:end_time]
            out_file_idx = idx+"_"+str(count).zfill(5)+"_f_"+str(start_time).zfill(5)+"_t_"+str(end_time).zfill(5)
            segment.export(out_dir+"/"+out_file_idx+".wav", format="wav")

            
            out_wav_file.write(out_file_idx+" "+out_dir+"/"+out_file_idx+".wav"+"\n")
            count += 1

            pass

        
        pass
    pass

