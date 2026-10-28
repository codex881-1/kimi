#!/usr/bin/env bash
# Frame-exact assembly of rendered parts + master audio (decode once, encode two deliverables).
#   part_00a [0,70)  part_00b [70,140)  part_0 [140,600) (trimmed)  part_0c [600,720)
#   part_0b [720,1047)  part_1 [1047,2094)
set -euo pipefail
cd "$(dirname "$0")/../out"
COMMON=(-color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart)
F="[2:v]trim=start_frame=140:end_frame=600,setpts=PTS-STARTPTS[c];"
for i in 0 1 3 4 5; do F+="[$i:v]setpts=PTS-STARTPTS[p$i];"; done
F+="[p0][p1][c][p3][p4][p5]concat=n=6:v=1:a=0,format=yuv420p,split=2[v1][v2]"
X264="aq-mode=3:psy-rd=1.0,0.15"
ffmpeg -v error -y -i part_00a.mp4 -i part_00b.mp4 -i part_0.mp4 -i part_0c.mp4 -i part_0b.mp4 -i part_1.mp4 -i master.wav \
  -filter_complex "$F" \
  -map "[v1]" -map 6:a -c:v libx264 -preset slow -crf 20 -maxrate 14M -bufsize 28M -profile:v high -level 4.2 \
     -x264-params "$X264" -c:a aac -b:a 256k -r 30 -shortest "${COMMON[@]}" heart_in_the_ice.mp4 \
  -map "[v2]" -map 6:a -c:v libx264 -preset slow -crf 17 -maxrate 40M -bufsize 80M -profile:v high -level 5.1 \
     -x264-params "$X264" -c:a aac -b:a 320k -r 30 -shortest "${COMMON[@]}" heart_in_the_ice_master.mp4
for f in heart_in_the_ice.mp4 heart_in_the_ice_master.mp4; do
  printf '%s  ' "$f"; ffprobe -v error -show_entries format=duration,size,bit_rate -of compact=p=0 "$f"
done
