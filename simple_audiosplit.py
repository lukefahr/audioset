#!/usr/bin/env python3

import os
import audiosplit

MY_DIR = os.path.dirname(os.path.realpath(__file__))
full_dir = MY_DIR + '/_data'

truck = {
    'label': 'truck',
    'includes': [ 'Truck', 'Medium engine (mid frequency)', ], 
    'excludes': [ 'Air brake', 
                'Air horn, truck horn', 
                'Reversing beeps',
                'Ice cream truck, ice cream van',
                'Fire engine, fire truck (siren)',
                'Jet engine',
                'Engine starting',
                'Accelerating, revving, vroom',
                'Car',
                'Wood',
                'Siren',
                'Toot',
              ],
    }
notruck = {
    'label': 'notruck',
    'includes' : [],
    'excludes' : ['Truck', 'Medium engine (mid frequency)', ],
    }

builder = (truck, notruck)
labels = list(map(lambda x: x['label'], builder))
includes = list(map(lambda x: x['includes'], builder))
excludes = list(map(lambda x: x['excludes'], builder))
num_clips = 800

#create the audio splitter
asp = audiosplit.AudioSplitter(
        data_dir = full_dir, sampling_rate = 16000, max_threads = 5,)

#run the full partial-clip (re)classification
data = asp.Run(labels, includes, excludes, num_clips = num_clips, clip_length_ms=500) 

# and write the resulting dataset to a csv file
data.to_csv(full_dir + '/dataset.csv', index=True, index_label='idx')


