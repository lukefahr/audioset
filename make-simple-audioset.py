#!/usr/bin/env python3
import audioset
import argparse

def get_truck_clips(N = 2 , sr=None, dirname=None):
    asb = audioset.AudioSetBuilder('balanced, unbalanced',sampling_rate=sr,data_dir=dirname)

    good_clips = asb.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                               excludes=[ 'Air brake',
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
                                          'Toot'
                               ], 
                               num_clips = N, download=True, max_threads=5) 
    print ('Good Clips: ')
    for clip in good_clips:
        print ('\t ' + str(clip))

def get_non_truck_clips(N = 2 , sr=None, dirname=None):
    asb = audioset.AudioSetBuilder('balanced, unbalanced',sampling_rate=sr,data_dir=dirname)
    
    bad_clips = asb.getClips ( includes = ['all'],
                               excludes =[ 'Truck', 'Medium engine (mid frequency)', 'Idling'],
                               num_clips = N, download=True, max_threads = 5)
    
    print ('Bad Clips: ')
    for clip in bad_clips:
        print ('\t ' + str(clip))


def main():
    
    def down_sampling_rate_type(x):
        x  = int(x)
        if x > 44100 or x < 5000:
            raise argparse.ArgumentTypeError("Sampling rate range 5000 - 44100")
        return x
        
    
    parser = argparse.ArgumentParser(description = 'Audio set builder. Downloads'+
                                     ' truck audio samples from Youtube\'s labelled dataset')
    parser.add_argument('download_dir', metavar='download_dir', type=str,\
                        help='Specifies the directory where the sound clips are downloaded to')

    parser.add_argument('ntrucks', metavar='ntrucks', type=int,\
                        help='# of truck samples',\
                        default=1)

    parser.add_argument('nnontrucks', metavar='nnontrucks', type=int,\
                        help='# of non-truck samples',\
                        default=1)

    
    parser.add_argument('-sr', metavar='down_sampling_rate', type=down_sampling_rate_type,\
                        help='Sampling rate of audio clips[Fetched clips will be downsampled]',\
                        default='14000', required=False)

    args = parser.parse_args()
    download_dir = args.download_dir
    ntrucks = args.ntrucks
    nnontrucks = args.nnontrucks
    sr = args.sr

    print('Passed sampling rate {}, download dir {}, #trucks {}, # non-trucks {}'.\
          format(sr, download_dir, ntrucks, nnontrucks))

    get_truck_clips(ntrucks, sr)
    get_non_truck_clips(nnontrucks, sr)
    
if __name__ == '__main__':
    main()
