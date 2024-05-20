for file in /data/Vad/silero-vad/15temps/wav/*.wav; do
    ffprobe -i "$file" -show_entries format=duration -v quiet -of csv="p=0"
done

