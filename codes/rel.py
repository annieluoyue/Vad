import os
import re
from pydub import AudioSegment
from utils_vad import *

# 定义输入和输出目录
wav_dir = "/data/BEA_wavs/release_1/sound"
out_dir = "/data/Vad/silero-vad/temps/wav/"

# 确保输出目录存在
os.makedirs(out_dir, exist_ok=True)

# 加载VAD模型
model_path = "your_model_path_here"  # 替换为您的模型路径
model = init_jit_model(model_path)

# 设置音频长度的最小值和最大值（单位：毫秒）
min_duration = 28000
max_duration = 32000

# 不符合命名要求的数字集合
excluded_numbers = set(["037", "070", "075", "102", "146", "154", "176", "196", "198", "463", ...])  # 省略其余数字

def process_audio(file_path, output_dir, min_dur, max_dur):
    audio = AudioSegment.from_file(file_path, format="wav")
    wav = read_audio(file_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

    concatenated_segment = AudioSegment.empty()
    concatenated_duration = 0
    count = 0

    for item in speech_timestamps:
        start_time = int(item['start'] / 16000 * 1000)
        end_time = int(item['end'] / 16000 * 1000)
        segment = audio[start_time:end_time]

        if concatenated_duration + len(segment) > max_dur:
            # 导出当前片段
            final_segment = concatenated_segment.set_frame_rate(16000).set_channels(1)
            final_segment.export(os.path.join(output_dir, f"{os.path.basename(file_path)}_{count}.wav"), format="wav")
            concatenated_segment = segment
            concatenated_duration = len(segment)
            count += 1
        else:
            concatenated_segment += segment
            concatenated_duration += len(segment)

    # 检查最后一段音频
    if concatenated_duration > 0:
        final_segment = concatenated_segment.set_frame_rate(16000).set_channels(1)
        final_segment.export(os.path.join(output_dir, f"{os.path.basename(file_path)}_{count}.wav"), format="wav")

# 处理每个文件
for filename in os.listdir(wav_dir):
    match = re.search(r'bea(\d{3})', filename)
    if match and match.group(1) not in excluded_numbers:
        process_audio(os.path.join(wav_dir, filename), out_dir, min_duration, max_duration)

# 验证输出音频长度
for out_file in os.listdir(out_dir):
    audio_length = len(AudioSegment.from_file(os.path.join(out_dir, out_file)))
    if audio_length < min_duration or audio_length > max_duration:
        print(f"文件 {out_file} 的长度 {audio_length} ms 不符合要求，正在重新处理...")
        # 对不符合要求的文件进行再处理

print("处理完成")

