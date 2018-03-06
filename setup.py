#!/usr/local/bin/python3

import logging 

import audioset

#
#
#
#
if __name__ == '__main__':
    
    logging.basicConfig ( level = logging.WARN,
                        format='%(levelname)s [0x%(process)x] %(name)s: %(message)s')
   

    gasd = audioset.GoodAudioSetData(2000, download=True, max_threads = 20, logLvl=logging.DEBUG)

    basd = audioset.BadAudioSetData(2000,  download=True, max_threads = 20, logLvl=logging.DEBUG)

