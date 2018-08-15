import os
import audioset

cwd = os.getcwd()
trucks    = cwd + '/data/trucks'
no_trucks = cwd + '/data/no_trucks'

audioset.GoodAudioSetData(num_samples=10, 
                          download=True,
                          framerate=22000, 
                          max_threads = 4,
                          data_dir=trucks)
audioset.BadAudioSetData(num_samples=10, 
                         download=True,
                         framerate=22000, 
                         max_threads = 4,
                         data_dir=no_trucks)
