#!/usr/bin/python3

'''
 This file is used to automatically download/convert YouTube clips give metadata
 from the AudioSet dataset. 

 It constists of:
    ClipDownloader:  Physically downloads the youtube data, and converting the 
                        audio to a wave file, this can be multi-threaded
    ClipFinder:  Scans the Audioset metadata files for clips matching your query
    AudioSetBuilder:  High-level API that automates finding/downloading clips. Also handles                         downsampling the audio if necessary. 
'''

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


#
#
#
#
#
class AudioSetBuilder (object):
    '''  Manages the finding, downloading, and downsampling of audio clips 
         from the AudioSet dataset
    '''
    #
    #
    #
    def __init__(this, framerate=22000, audioset='balanced',
                eval_file = None, balanced_file = None, unbalanced_file = None, \
                ontol_file = None,\
                data_dir=None, logLvl=logging.INFO):

        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
        this.log.info ("Created")
        
        this.mydir = os.path.dirname(os.path.realpath(__file__))

        this.ddir = this.mydir+'/_data' if data_dir  == None \
                        else data_dir

        this.framerate = framerate
        assert( this.framerate > 0 and this.framerate < 1e6)
        
        audioset = 'eval,balanced,unbalanced' if audioset == None \
                            else audioset
        assert ('eval' in audioset or 'balanced' in audioset)                            

        this.cf = ClipFinder( audioset = audioset, 
                        eval_file=eval_file, balanced_file=balanced_file, 
                        unbalanced_file=unbalanced_file, ontol_file=ontol_file,
                        logLvl=logLvl)
        this.cdl = ClipDownloader( data_dir = this.ddir , 
                        logLvl=logLvl) 


    #
    #
    #
    def getClips (this, includes, excludes=[], num_clips=1,  
                        download=False, max_threads = 1):
        ''' does all the work to collect and gather clips, returing a 
            list of file names'''

        if isinstance(includes, str): includes= [includes]
        if isinstance(excludes, str): excludes= [excludes]

        # no metas, that's fine
        if num_clips == 0: return
        assert( num_clips >0)
        assert( len(includes) > 0 ) 

        this.metas = []
        
        # this is a bit hacky, but try progressively larger searches 
        # until we've found enough *working* clips for our needs
        meta_num = num_clips

        for i in range(5,0,-1): # try at most 5 round of expanded searches

            this.log.debug('collecting ' + str(meta_num) + ' metadatas')

            raw_metas = this.cf.search( includes, excludes, meta_num) 
            if len(raw_metas) < meta_num:
                this.log.warn('insufficient clips available: ' + 
                            str(len(raw_metas)) )
                break

            if download:
                this.log.debug('downloading clips')
                this.cdl.download(raw_metas, max_threads = max_threads)

            this.metas = this._verifyMetas(raw_metas)

            if len(this.metas) < num_clips:
                this.log.debug('some metas not downloaded')
                this.log.debug('have: ' + str(len(this.metas)) + 
                                    ' want: ' + str(num_clips))
                if i > 1: this.log.debug(' trying bigger search')
                else: this.log.debug(' giving up')
                meta_num *= 2
            else:
                this.log.debug('found sufficient metas')
                break

        this.log.debug('formatting clip names')
        this.metas = this.metas[:num_clips]
        this.clips  = list(map(lambda x: this.ddir + '/'  + x[0] + '.wav', this.metas))

        return this.clips


    #def _getSampleData(this, ytid):
    #    
    #    this.log.info('Loading ytid: ' + str(ytid))

    #    fname = this.ddir + '/' + ytid + '.wav'
    #    assert( os.path.exists(fname))
    #   
    #    rate = 'r_' + str(int(this.framerate)) 
    #    fname_rate = this.ddir + '/' + \
    #            rate + '/' + ytid + '.wav'
    #    if not os.path.exists(fname_rate):
    #        this._downsample(fname, this.framerate, fname_rate)
    #    
    #    aud = pydub.AudioSegment.from_wav( fname_rate)
    #    aud = aud.set_channels(1)
    #    data = aud.get_array_of_samples()
    #    data = np.asarray(data)

    #    return AudioData (data, this.framerate, ytid)


    def _downsample(this, fname, new_rate, new_fname):
        assert( os.path.exists(fname) )
        
        dirname = os.path.dirname(new_fname)
        if not os.path.exists(dirname):
            this.log.debug('Creating: ' + str(dirname))
            os.makedirs(dirname)

        this.log.debug('Downsampling to: ' + str(new_rate))

        down = pydub.AudioSegment.from_wav( fname)
        down = down.set_frame_rate(new_rate)
        down = down.set_channels(1)
        _ = down.export( new_fname, format='wav')


    #def _downloadYTIDs(this, ytids, threads=1): 
    #    ''' download a list of ytids into the unified folder '''
    #    
    #    this.log.debug('downloading ytids')
    #    
    #    if isinstance(ytids, str): ytids = [ytids]

    #    exclude = []
    #    for ytid in ytids:
    #        #if it's already there, skip downloading
    #        if os.path.exists( this.ddir + '/' + str(ytid) + '.wav'):
    #            this.log.debug('ytid: ' + str(ytid) + ' already exists')
    #            exclude += [ytid]
    #    
    #    #this is not that efficient...
    #    ytids = [x for x in ytids if x not in exclude ]

    #    if len(ytids) == 0:
    #        this.log.debug('all metas already present')
    #        return

    #    this.log.debug('gathering metadatas for ytids')
    #    cf = clipFind.ClipFinder( audioset='eval,balanced,unbalanced', 
    #                                logLvl=this.log.getEffectiveLevel())
    #    metas = cf.searchByYTIDs(ytids)

    #    return this._downloadMetas(metas, threads)

    def _getMetasByYTIDs(this, ytids):
        this.log.debug('collecting metas for random ytids')
        cf = clipFind.ClipFinder( audioset='eval,balanced,unbalanced', 
                                    logLvl=this.log.getEffectiveLevel())
        return cf.searchByYTIDs(ytids)

    #
    #
    #
    def _verifyMetas(this, metas):
        ''' double-check that all the files exists on disk '''
        ret_metas = []

        for meta in metas:
            ytid= meta[0]
            
            if os.path.exists(this.ddir + '/' + ytid + '.wav'):
                ret_metas.append( meta)

        return ret_metas




#
#
#
#
#
class ClipFinder:
    ''' 
        scans the Audioset Metadata files for clips matching the specifications
        and returns their metadata files
    '''

    class BreakException(Exception): pass
   
    #
    #
    #
    def __init__(this, audioset='balanced,unbalanced',  \
                eval_file = None, balanced_file = None, unbalanced_file = None, \
                ontol_file = None,\
                logLvl=logging.INFO):
        
        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
        this.log.info ("Created")
       
        this.mydir = os.path.dirname(os.path.realpath(__file__))
     
        if eval_file == None:  
            eval_file = this.mydir + '/eval_segments.csv'
        if balanced_file == None: 
            balanced_file = this.mydir + '/balanced_train_segments.csv' 
        if unbalanced_file == None:
            unbalanced_file = this.mydir + '/unbalanced_train_segments.csv'

        audioset = audioset.lower().split(',')
        this.audiosets = []
        if 'eval' in audioset:
            this.audiosets += [ eval_file] 
        if 'balanced' in audioset:
            this.audiosets += [ balanced_file ]
        if 'unbalanced' in audioset:
            this.audiosets += [ unbalanced_file ] 

        for audset in this.audiosets:
            assert( os.path.exists(audset) )
        
        if ontol_file == None: this.ontol = this.mydir + '/ontology/ontology.json'
        else: this.ontol = ontol_file
        assert( os.path.exists(this.ontol))
        this.log.info('opening ontology json: ' + str(this.ontol))
        this.ontol = open(this.ontol).read()
        this.ontol = json.loads(this.ontol)


    def search( this, includes=[], excludes=[], max_clips=1):
        ''' returns a number of (youtubeID, start_time, end_time) tuples 
            for a given set of human readable ontology names 
        '''

        assert( max_clips >0)
        assert( len(includes) > 0 ) 

        this.log.debug('Searching ' + str(this.audiosets)  \
                    + 'for ' + str(includes) + ' includes, and ' \
                    + str(excludes) + ' excludes.')

        # translate ontology names to labels
        include_labels = this._getIDbyNames( includes ) 
        exclude_labels = this._getIDbyNames( excludes ) 
       
        if len(include_labels) == 0: 
            this.log.debug('Including ALL labels')
            include_labels = ['all']
 
        # then continue the search
        return this._searchByLabels( include_labels, exclude_labels, max_clips)

    #
    #
    #
    def _searchByLabels( this, includes=[], excludes=[], max_clips=1):
        ''' returns a series of (ytid,start,end) metadatas
            given labels
        '''
        clips = []

        # in case we need to end early
        try: 
            # loop over all audiosets
            for audsetf in this.audiosets:
                
                audset = this._openAudSet(audsetf)

                # loop over all rows
                for row in audset:
                    
                    #default to add
                    add = 'all' in includes
                    
                    # first try to include the clip if possible
                    if add == False:
                        
                        for label in includes:
                            if label in row['positive_labels']: 
                                this.log.debug('Including ' + row['YTID'] + \
                                        ' (' + label + ')')
                                add= True
                                break

                    # if we still think add, check the clip against
                    # the exclude list
                    if add == True:
                        for label in excludes:
                            if label in row['positive_labels']: 
                                this.log.debug('Excluding ' + row['YTID'] + \
                                        ' (' + label + ')')
                                add= False
                                break
                    
                    #if the clip makes it past exclude, then we're good
                    if add == True:
                        this.log.debug('Adding ' + row['YTID'] ) 
                        clips.append( (row['YTID'],row['start_seconds'],
                                        row['end_seconds'] ) )
                    
                    #stop early if we have enough clips
                    if max_clips!=None and len(clips) >= max_clips:
                        this.log.debug('Found sufficient number of clips found (' 
                                        + str(len(clips)) + '), stopping')
                        raise this.BreakException            
                
        except this.BreakException: pass

        this.log.debug('Total Clips : ' + str(len(clips)))

        return clips                        

    def searchByYTIDs(this, ytids):
        ''' return a (youtubeID, start_time, end_time) tuple for the 
            given set of ytids, mainly used for testing
        '''
        if isinstance(ytids, str):
            ytids = [ytids]

        clips = []
    
        try:
            for audsetf in this.audiosets:
                audset = this._openAudSet(audsetf)

                for row in audset:
                    
                    for ytid in ytids:

                        if row['YTID'] == ytid:
                            clips.append( (row['YTID'],row['start_seconds'],
                                            row['end_seconds'] ) )
                        
                        if len(clips) == len(ytids):
                            this.log.debug('found all clips')
                            raise this.BreakException            

        except this.BreakException: pass

        return clips

    #def nameLookup( this, ytid):
    #    ''' 
    #    '''

    #    this.log.debug('Looking up names for ytid: ' + str(ytid))

    #    for audsetf in this.audiosets:
    #        audset = this._openAudSet(audsetf)

    #        for row in audset: 
    #            
    #            if row['YTID'] == ytid:
    #                this.log.debug('found ytid: ' + str(row))
    #                labels = row['positive_labels']
    #                this.log.debug('found labels: ' + str(labels))
    #                names = [ x['name'] for x in this.ontol if x['id'] in labels]
    #                this.log.debug('found names: ' + str(names))
    #                
    #                return names
    #    return None


    #
    #
    #
    def _getIDbyNames(this, names):
        ''' selects ontology ID's by their corresponding human readable ontology names'''
        
        this.log.debug('Selecting Ontology ID by NAME(S): ' + str(names))
        if isinstance(names, str): names = [ names ]

        subs = [ x for x in this.ontol if x['name'] in names ]
        
        return [ x['id'] for x in subs ]

    #
    #
    #
    def _openAudSet(this, fname):
        ''' opens an audioset file and parses it's contents as a dictionary'''
        audset = open(fname, 'r')

        # the first few lines are not relivant
        this.log.debug( 'skipping: ' + str(audset.readline() ))
        this.log.debug( 'skipping: ' + str(audset.readline() ))
        this.log.debug( 'skipping: ' + str(audset.readline() ))

        audset= csv.DictReader(audset.readlines(), 
                        fieldnames=[ 'YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                        delimiter=',',
                        quotechar='"',
                        quoting = csv.QUOTE_ALL,
                        skipinitialspace=True)
        
        return audset


#
#
#
#
#
class ClipDownloader(object):
    ''' downloader for youtube clips '''
    
    def __init__(this, data_dir=None, logLvl= logging.WARN):

        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
        this.log.info ("Created")

        this.mydir = os.path.dirname(os.path.realpath(__file__))
        this.ddir =  data_dir
        if this.ddir == None:
            this.ddir = this.mydir + '/_data'

        if not os.path.exists(this.ddir):
            this.log.debug('Creating data directory: ' + str(this.ddir))
            os.makedirs(this.ddir)

    def download(this, metas, max_threads = 1):
        ''' splits up the list of metas, begins downloading the clips '''
        
        assert( max_threads > 0)
        assert( len(metas) > 0)

        this.log.debug('downloading metas with max_threads: ' + str(max_threads))

        if isinstance(metas, tuple):
            this.log.debub('found tuple, promoting to list')
            metas = [ metas ] 

        this.log.info('Collecting' + str(len(metas)) + ' metadatas')
   
        if max_threads == 1:
            this.log.info('Single-Threaded Clip Downloader')
            this._download_st( metas, this.ddir)
        else:
            this.log.info('Multi-Threaded Clip Downloader')
            this._download_mt(metas, this.ddir, max_threads)

        this.log.info('downloading complete!')

    
    #
    #
    #
    def _download_mt(this, metas, data_dir, max_threads):
        ''' Multi threaded meta downloader '''

        this.log.debug('Multi Threaded Clip Downloader')

        threads = []

        # this really should launch 1 process for a subset of the list of metas
        # rather than a seperate process for each meta

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
                        target=this._download_st, \
                        kwargs = dict(metas=[meta], data_dir=data_dir))

            this.log.debug('Launching Thread for: ' + str(meta[0]))
            t.start()
            threads.append(t)
            time.sleep(0.01)

        this.log.debug('Waiting for last threads to finish')
        threads = list(filter( lambda x: x.exitcode == None, threads))
        for t in threads:
            t.join()

        this.log.debug('done downloading')

    #
    #
    #
    def _download_st(this, metas, data_dir):
        ''' Single threaded meta downloader '''
        
        this.log.debug('Single Threaded Clip Downloader')

        for meta in metas:

            cid, cstart, cend  = meta
            this.log.debug('downloading clip: ' + str(cid) + 
                            ' s: ' + str(cstart) + ' e:' + str(cend))

            this._download_clip(
                    ytid=cid, start=cstart, stop=cend, data_dir=data_dir)
        

        this.log.debug('done downloading')

    
    #
    #
    #
    def _download_clip(this, ytid, data_dir, start=0.0, stop=None):
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
            try:
                origwd = os.getcwd()
            except FileNotFoundError:
                this.log.info('Bad Dir? Using %s' % this.mydir)
                origwd = this.mydir

            with tempfile.TemporaryDirectory( 
                        suffix=str(threading.get_ident())) as tmpdir:
                os.chdir(tmpdir)
            
                tmpname =  this._download_ytid(ytid)
                if tmpname != None:
                    this._crop_clip( tmpname, fname, start, stop)

            os.chdir(origwd)

    #
    #
    #
    def _download_ytid(this, ytid):
        '''
        Actually downloads a youtube video, extracts the raw audio
        NOTE:  downloads to current working directory, so chdir first!
        Returns NONE if the video is not valid
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
        try: 
            return os.getcwd() + '/' + ytid + '.wav'
        except:
            return None

    #
    #
    #
    def _crop_clip(this, in_fname, out_fname, start=0, stop=None): 
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
        



#
#
#
#
#
def TestClipDownloader():

    ''' tests for the ClipDownloader class '''

    testdir = os.getcwd() + '/__testClipDownloader' 
    
    if os.path.exists(testdir): shutil.rmtree(testdir)

    cdl = ClipDownloader( data_dir = testdir, logLvl=logging.DEBUG)

    metas = [('-2PDE7hUArE', '30.000', '40.000'), 
            ('-DNkAalo7og', '30.000', '40.000'), 
            ('-GDC7PuqdOM', '30.000', '40.000'), 
            ('-jBWkHhQNew', '30.000', '40.000'), 
            ('-x2aAKUtNRw', '30.000', '40.000')] 
    cdl.download(metas[0:2], max_threads = 1)
    cdl.download(metas, max_threads = 5)


#
#
#
def TestClipFinder():
    ''' tests for the ClipFinder class '''
   
    dg = ClipFinder( logLvl=logging.DEBUG)
    clips = dg.search( includes=['Truck',
                                'Medium engine (mid frequency)', 
                                ], 
                    excludes=[ 'Air brake', 
                                'Air horn, truck horn', 
                                'Reversing beeps', 
                                'Ice cream truck, ice cream van', 
                                'Fire engine, fire truck (siren)',
                                'Jet engine', 
                               ], 
                    max_clips = 5) 
    clip_ytids = [ x[0] for x in clips ]

    clips2 = dg.searchByYTIDs(clip_ytids)

    for x,y in zip(clips, clips2):
        assert( x==y)

    print (clips)
    print (clips2)

    dg = ClipFinder( audioset='eval,balanced,unbalanced', logLvl=logging.DEBUG)
    clip = dg.searchByYTIDs('zfLqqw47CrM')

    assert( clip[0] == ('zfLqqw47CrM', '520.000', '530.000') )
    print (clip)

#
#
#
#
#
def TestAudioSetBuilder( delete=True):

    ''' tests for the AudioSetBuilder class '''

    testdir = os.getcwd() + '/__testAudioSetBuilder' 
    
    if delete:
        if os.path.exists(testdir): shutil.rmtree(testdir)

    print ('TESTING')
    asd = AudioSetBuilder( data_dir=testdir, logLvl=logging.DEBUG)
    clips = asd.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                  excludes=[ 'Air brake', 
                                'Air horn, truck horn', 
                                'Reversing beeps', 
                                'Ice cream truck, ice cream van', 
                                'Fire engine, fire truck (siren)',
                                'Jet engine', 
                               ], 
                    num_clips = 2, download=True, max_threads=1) 
    ref_clips = [ '-2PDE7hUArE.wav', '-DNkAalo7og.wav']
    ref_clips = list(map(lambda x: testdir + '/' + x, ref_clips))
    assert( all([x==y for x, y in zip(clips, ref_clips)]))
    assert(len(clips) > 0)

    print ('TESTING, download=False')
    asd = AudioSetBuilder( data_dir=testdir, logLvl=logging.DEBUG)
    clips = asd.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                  excludes=[ 'Air brake', 
                                'Air horn, truck horn', 
                                'Reversing beeps', 
                                'Ice cream truck, ice cream van', 
                                'Fire engine, fire truck (siren)',
                                'Jet engine', 
                               ], 
                    num_clips = 2, download=False, max_threads=1) 
    ref_clips = [ '-2PDE7hUArE.wav', '-DNkAalo7og.wav']
    ref_clips = list(map(lambda x: testdir + '/' + x, ref_clips))
    assert( all([x==y for x, y in zip(clips, ref_clips)]))
    assert(len(clips) > 0)

    print ('TESTING, specifying eval audioset, download=False')
    asd = AudioSetBuilder( data_dir=testdir, logLvl=logging.DEBUG,
                            audioset='eval')
    clips = asd.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                  excludes=[ 'Air brake', 
                                'Air horn, truck horn', 
                                'Reversing beeps', 
                                'Ice cream truck, ice cream van', 
                                'Fire engine, fire truck (siren)',
                                'Jet engine', 
                               ], 
                    num_clips = 2, download=False, max_threads=1) 
    if delete:
        assert(len(clips) == 0)

    print ('TESTING, specifying eval audioset')
    asd = AudioSetBuilder( data_dir=testdir, logLvl=logging.DEBUG,
                            audioset='eval')
    clips = asd.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                  excludes=[ 'Air brake', 
                                'Air horn, truck horn', 
                                'Reversing beeps', 
                                'Ice cream truck, ice cream van', 
                                'Fire engine, fire truck (siren)',
                                'Jet engine', 
                               ], 
                    num_clips = 2, download=True, max_threads=1) 
    ref_clips = [ '-BY64_p-vtM.wav', '-HWygXWSNRA.wav']
    ref_clips = list(map(lambda x: testdir + '/' + x, ref_clips))
    assert( all([x==y for x, y in zip(clips, ref_clips)]))


    print ('TESTING, specifying specific files')
    mydir = os.path.dirname(os.path.realpath(__file__))
    asd = AudioSetBuilder( data_dir=testdir, logLvl=logging.DEBUG,
            eval_file = mydir + '/eval_segments.csv',
            balanced_file = mydir + '/balanced_train_segments.csv',
            unbalanced_file = mydir + '/unbalanced_train_segments.csv',
            ontol_file = mydir + '/ontology/ontology.json',
            ) 
    clips = asd.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                  excludes=[ 'Air brake', 
                                'Air horn, truck horn', 
                                'Reversing beeps', 
                                'Ice cream truck, ice cream van', 
                                'Fire engine, fire truck (siren)',
                                'Jet engine', 
                               ], 
                    num_clips = 3, download=True, max_threads=5) 
    ref_clips = ['-2PDE7hUArE.wav', '-DNkAalo7og.wav', '-GDC7PuqdOM.wav']
    ref_clips = list(map(lambda x: testdir + '/' + x, ref_clips))
    assert( all([x==y for x, y in zip(clips, ref_clips)]))




#
#
#
#
#
if __name__ == '__main__':

    logging.basicConfig ( level = logging.WARN,
                        format='%(levelname)s [0x%(process)x] %(name)s: %(message)s')

    TestClipDownloader()
    TestClipFinder()
    TestAudioSetBuilder( delete=True)

    print ('TESTING PASSED')
