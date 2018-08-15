import os
import audioset

cwd = os.getcwd()
trucks    = cwd + '/data/trucks'
no_trucks = cwd + '/data/no_trucks'

asb = audioset.AudioSetBuilder()

good_clips = asb.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                  excludes=[ 'Air brake', 
                                'Air horn, truck horn', 
                                'Reversing beeps', 
                                'Ice cream truck, ice cream van', 
                                'Fire engine, fire truck (siren)',
                                'Jet engine', 
                               ], 
                    num_clips = 10, download=True, max_threads=5) 
print ('Good Clips: ')
for clip in good_clips:
    print ('\t ' + str(clip))


bad_clips = asb.getClips ( includes = ['all'],
                excludes =[ 'Truck', 'Medium engine (mid frequency)', 'Idling'],
                num_clips = 10, download=True, max_threads = 5)

print ('Bad Clips: ')
for clip in bad_clips:
    print ('\t ' + str(clip))
