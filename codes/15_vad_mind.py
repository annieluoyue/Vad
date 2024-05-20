import os
import re
import sys
from pydub import AudioSegment
from utils_vad import *

# 定义输入和输出目录
wav_dir = "/data/BEA_wavs/mind"

out_dir = "/data/Vad/silero-vad/15temp/wav/"

# 加载模型

# 设置音频长度的最小值和最大值（单位：毫秒）
min_duration = 15000
max_duration = 35000

model_path = sys.argv[1]
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

        for item in speech_timestamps:
            start_time = int(item['start'] / 16000 * 1000)
            end_time = int(item['end'] / 16000 * 1000)
            segment = audio[start_time:end_time]
            segment_duration = end_time - start_time

            while segment_duration > 0:
                if concatenated_duration + segment_duration >= min_duration:
                    # 裁剪这个片段以满足时长要求
                    trim_length = min(segment_duration, max_duration - concatenated_duration)
                    segment_to_add = segment[:trim_length]
                    segment = segment[trim_length:]
                    segment_duration -= trim_length
                    concatenated_duration += trim_length
                    concatenated_segment += segment_to_add

                    if concatenated_duration >= min_duration:
                        # 导出音频并重置拼接段
                        out_file_idx = filename.split(".")[0] + "_" + str(count).zfill(5) + "_concat"
                        final_segment = concatenated_segment.set_frame_rate(16000).set_channels(1)
                        final_segment.export(os.path.join(out_dir, out_file_idx + ".wav"), format="wav")
                        concatenated_segment = AudioSegment.empty()
                        concatenated_duration = 0
                        count += 1
                else:
                    break

print("处理完成")
