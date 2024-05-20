#!/bin/bash

total_duration=0

for file in /data/Vad/silero-vad/release_1_cut/wav/*.wav; do
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    total_duration=$(echo "$total_duration + $duration" | bc)
done

echo "Total duration: $total_duration seconds"

