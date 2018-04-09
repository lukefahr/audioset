# Truck Audio Set Generator

This project used youtube and google's ontology to download
an audio set of truck versus non-truck sounds. For basic
usage use `make-simple-audioset.py` which is an example
that downloads 10 truck sounds and 10 non-truck sounds.

Google rate limits downloads from youtube but parallel downloads
via multiple threads significantly improves download speeds. This
is the purpose of the `max_threads` parameter. 

## Dependencies

This project requires python 3 and ffmpeg in addition to the
python packages youtube-dl, numpy, and pydub.


## try this to get started: 

```bash
pip3 install youtube_dl pydub numpy

git clone https://github.com/lukefahr/audioset.git
cd audioset
git submodule update --init
python3 make-simple-audioset.py
```

