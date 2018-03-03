#!/usr/local/bin/python3

import csv
import logging
import itertools
import json
import multiprocessing
import os
import pydub 
import shutil
import tempfile
import threading
import time
import random
import youtube_dl


class ClipDownloader(object):
    ''' downloader for youtube clips '''
    
    def __init__(this, data_dir=None, logLvl= logging.WARN):

        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)

        this.log.info ("Created")

        this.mydir = os.path.dirname(os.path.realpath(__file__))
        this.ddir =  data_dir
        if this.ddir == None:
            this.ddir = this.mydir + '/_unified_data'

        if not os.path.exists(this.ddir):
            this.log.debug('Creating data directory: ' + str(this.ddir))
            os.makedirs(this.ddir)

    def download(this, metas, max_threads = 1):
        
        assert( max_threads > 0)
        assert( len(metas) > 0)

        this.log.debug('downloading metas with max_threads: ' + str(max_threads))

        if isinstance(metas, tuple):
            this.log.debub('found tuple, promoting to list')
            metas = [ metas ] 

        this.log.info('Collecting' + str(len(metas)) + ' metadatas')
   
        if max_threads == 1:
            this.log.info('Single Threaded Clip Gatherer')
            this._gather_st( metas, this.ddir)
        else:
            this.log.info('Multi Threaded Clip Gatherer')
            this._gather_mt(metas, this.ddir, max_threads)

        this.log.info('Gathering complete!')

    #
    #
    #
    def _gather_st(this, metas, data_dir):
        ''' Single threaded meta gatherer '''
        
        this.log.debug('Single Threaded Clip Gatherer')

        for meta in metas:

            cid, cstart, cend  = meta
            this.log.debug('Gathering clip: ' + str(cid) + 
                            ' s: ' + str(cstart) + ' e:' + str(cend))

            this._build_ytid(
                    ytid=cid, start=cstart, stop=cend, data_dir=data_dir)
        

        this.log.debug('done gathering')


    #
    #
    #
    def _gather_mt(this, metas, data_dir, max_threads):
        ''' Multi threaded meta gatherer '''

        this.log.debug('Multi Threaded Clip Gatherer')

        threads = []

        for meta in metas:                
            
            # do a check to prevent launching a process over nothing
            if os.path.exists( data_dir + '/' + meta[0] + '.wav'):
                this.log.debug('YTID exists: ' + str(meta[0]))
                continue

            #stop if thread limited
            while len(threads) >= max_threads:
                threads = list(filter( lambda x: x.exitcode == None, threads))
                if len(threads) >= max_threads: time.sleep(0.1)
               
            #t = threading.Thread(
            t = multiprocessing.Process(
                        target=this._gather_st, \
                        kwargs = dict(metas=[meta], data_dir=data_dir))

            this.log.debug('Launching Thread for: ' + str(meta[0]))
            t.start()
            threads.append(t)

        this.log.debug('Waiting for last threads to finish')
        threads = list(filter( lambda x: x.exitcode == None, threads))
        for t in threads:
            t.join()

        this.log.debug('done gathering')

   
    #def _findClipsByYTID(this, ytid):
    #    ''' return a (youtubeID, start_time, end_time) tuple for the 
    #        given ytid
    #    '''
    #    for row in this.audset:
    #        if row['YTID'] == ytid:
    #            return (row['YTID'],row['start_seconds'],row['end_seconds'] ) 

    #    return None

    def _build_ytid(this, ytid, data_dir, start=0.0, stop=None):
        '''
            Gets a youtube video based on youtube id, pulls out
            the audio between start and stop, and saves it to the data
            directory
        '''
         
        ddir = data_dir 

        this.log.info('Downloading: ' + str(ytid) )
        this.log.debug('\t\t' + ' into ' + str(ddir))

        fname = ddir + '/' + ytid + '.wav'

        if os.path.exists(fname):
            this.log.info('File already exists: ' + str(ytid) + '.wav. Skipping.')

        else: 
            origwd = os.getcwd()
            with tempfile.TemporaryDirectory( 
                        suffix=str(threading.get_ident())) as tmpdir:
                os.chdir(tmpdir)
            
                tmpname = this._download_ytid(ytid)
                if tmpname != None:
                    this._crop( tmpname, fname, start, stop)

            os.chdir(origwd)

    def _crop(this, in_fname, out_fname, start=0, stop=None): 
        ''' 
        Crops a wave file on disk, overwrites the current file
        '''

        this.log.debug('cropping : ' + str(in_fname) )
        
        try:
            orig_aud = pydub.AudioSegment.from_wav(in_fname)
        except Exception as e:
            this.log.ERR('Error occured while loading audiofile: ' + 
                            str(e))
            return                            

        if isinstance(start, str): start = int(float(start))
        start_ms = int(start * 1000)

        if stop:
            if isinstance(stop, str): stop= int(float(stop))
            stop_ms = int(stop * 1000)
            new_aud = orig_aud[start_ms:stop_ms]
        else:
            new_aud = orig_aud[start_ms:]
        
        this.log.debug('writing to: ' + str(out_fname))
        new_aud.export(out_fname, format='wav')
        

    def _download_ytid(this, ytid):
        '''
        Downloads a youtube video, extracts wav audio
        NOTE:  downloads to current working directory, so chdir first!
        '''
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'logger': this.log,
            #'progress_hooks': [my_hook],
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            url = 'https://www.youtube.com/watch?v=' + str(ytid)
            this.log.debug("Downloading YTID: " + str(ytid) + ' : ' + str(url))
            try:
                ydl.download([url] )
            except youtube_dl.utils.DownloadError as e:
                this.log.warn('Video not found, skipping ' + str(ytid))
                return None

        return os.getcwd() + '/' + ytid + '.wav'

      
#
#
#
#
#
if __name__ == '__main__':
    
    logging.basicConfig ( level = logging.WARN,
                        format='%(levelname)s [0x%(process)x] %(name)s: %(message)s')
    
    testdir = os.getcwd() + '/__test' 
    
    if os.path.exists(testdir):
        shutil.rmtree(testdir)

    cdl = ClipDownloader( data_dir = testdir, logLvl=logging.DEBUG)

    metas = [('-2PDE7hUArE', '30.000', '40.000'), 
            ('-DNkAalo7og', '30.000', '40.000'), 
            ('-GDC7PuqdOM', '30.000', '40.000'), 
            ('-jBWkHhQNew', '30.000', '40.000'), 
            ('-x2aAKUtNRw', '30.000', '40.000')] 
    cdl.download(metas[0:2], max_threads = 1)
    cdl.download(metas, max_threads = 5)


