PYTHON := .venv/bin/python

.PHONY: audio picture mux verify stills

audio:
	$(PYTHON) src/audio.py

picture:
	mkdir -p build
	$(PYTHON) src/render.py --video --fps 60 --scale 1 --output build/picture.mp4

mux:
	mkdir -p dist
	ffmpeg -hide_banner -loglevel warning -y -i build/picture.mp4 -i assets/audio/mix.wav \
		-c:v copy -c:a aac -b:a 256k -ar 48000 -ac 2 -t 68 -movflags +faststart dist/senior-bowl.mp4

stills:
	mkdir -p build/review
	$(PYTHON) src/render.py --stills 1.8,10.8,20,24.8,32.5,36.8,39.2,42.5,45.7,49.5,53.5,60.8,66.5 \
		--scale 0.5 --output build/review/frame

verify:
	ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
		-of json dist/senior-bowl.mp4
