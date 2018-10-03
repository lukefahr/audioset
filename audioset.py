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
import shutil
import subprocess
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
    def __init__(this, audioset='balanced',
                eval_file = None, balanced_file = None, unbalanced_file = None, \
                ontol_file = None,\
                data_dir=None, logLvl=logging.INFO):

        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
        this.log.info ("Created")
        
        this.mydir = os.path.dirname(os.path.realpath(__file__))

        this.ddir = this.mydir+'/_data' if data_dir  == None \
                        else data_dir

        
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
    def getClips (this, num_clips, includes=['all'], excludes=[],
                        download=False, max_threads = 1):
        ''' does all the work to collect and gather clips, returing a 
            list of file names'''

        # no clips, that's fine
        if num_clips == 0: return
        # but no negative clips
        assert( num_clips >0)

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
        this.clips  = list(map(lambda x: this.ddir + '/'  + x['YTID'] + '.wav', this.metas))

        return this.clips

    #
    #
    #
    def info(this, fnames):
        ''' returns the metadata and labels for a given file'''
        
        if isinstance(fnames, str):
            fnames = [fnames]
        
        ytids = list(map(lambda x: x.split('/')[-1][:-4], fnames))
        
        metas = this.cf.searchByYTIDs(ytids)
        
        for meta in metas: 
            meta['label_names'] = []
            for labelID in meta['positive_labels'].split(','):
                meta['label_names'] += [this.cf.labelName(labelID)]

        if len( metas) == 1:
            return metas[0]
        else: return metas

        
 

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
            ytid= meta['YTID']
            
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
        this.log.info ("Created ")
       
        this.mydir = os.path.dirname(os.path.realpath(__file__))
     
        if eval_file == None:  
            eval_file = this.mydir + '/eval_segments.csv'
        if balanced_file == None: 
            balanced_file = this.mydir + '/balanced_train_segments.csv' 
        if unbalanced_file == None:
            unbalanced_file = this.mydir + '/unbalanced_train_segments.csv'

        this.audiosets = []
        for audioset in audioset.lower().split(','):
            if 'eval' == audioset:
                this.log.debug('adding: ' + str(eval_file))
                this.audiosets += [ eval_file] 
            elif 'balanced' == audioset:
                this.log.debug('adding: ' + str(balanced_file))
                this.audiosets += [ balanced_file ]
            elif 'unbalanced' == audioset:
                this.log.debug('adding: ' + str(unbalanced_file))
                this.audiosets += [ unbalanced_file ] 
            else:   raise Exception("Unrecognized audioset")

        for audset in this.audiosets:
            assert( os.path.exists(audset) )

        this.log.info("Total Audiosets: " + str(this.audiosets) )
        
        if ontol_file == None: this.ontol = this.mydir + '/ontology/ontology.json'
        else: this.ontol = ontol_file
        assert( os.path.exists(this.ontol))
        this.log.info('opening ontology json: ' + str(this.ontol))
        this.ontol = open(this.ontol, 'rb' ) #binary keeps Py 3.6.5 happy
        this.ontol = json.load(this.ontol)


    def search( this, includes=['all'], excludes=[], max_clips=1):
        ''' returns an ordered dictionary of 
            ['YTID', start_time, end_time, positive_labels] 
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
        ''' returns a series of (ytid,start,end,labels) metadatas
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
                        clips.append( row )                     

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


        listYtids = list(ytids)
        setYtids = set(ytids)

        this.log.debug('search for ytids: ' + str (listYtids[:4]) + 
                        ('...' if len(listYtids)>4 else '') )

        dictClips = {}
    
        try:
            for audsetf in this.audiosets:
                audset = this._openAudSet(audsetf)

                for row in audset:
                    
                    if row['YTID'] in setYtids:
                        this.log.debug('found ytid: ' + row['YTID'])

                        dictClips[row['YTID']] = row  
                           
                        if len(dictClips) == len(listYtids):
                            this.log.debug('found all clips')
                            raise this.BreakException            

        except this.BreakException: pass
        
        # have unordered clips, need to make list
        listClips = [ dictClips[x] for x in listYtids]

        return listClips

    
    def labelName( this, labelID):
        
        this.log.debug('Looking up names for labelID: ' + str(labelID))
        
        for line in this.ontol:
            if line['id'] == labelID:
                return line['name']
        
        this.log.debug('No names found for labelID: ' + str(labelID))
        return None

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
            if os.path.exists( data_dir + '/' + meta['YTID'] + '.wav'):
                this.log.debug('YTID exists: ' + str(meta['YTID']))
                continue

            #stop if thread limited
            while len(threads) >= max_threads:
                threads = list(filter( lambda x: x.exitcode == None, threads))
                if len(threads) >= max_threads: time.sleep(0.1)
               
            #t = threading.Thread(
            t = multiprocessing.Process(
                        target=this._download_st, \
                        kwargs = dict(metas=[meta], data_dir=data_dir))

            this.log.debug('Launching Thread for: ' + str(meta['YTID']))
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
            this.log.debug('downloading clip: ' + str(meta['YTID']) + 
                            ' s: ' + str(meta['start_seconds']) + ' e:' + str(meta['end_seconds']))

            this._download_clip(
                    ytid=meta['YTID'], start=meta['start_seconds'], stop=meta['end_seconds'], data_dir=data_dir)
        

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
       
        if isinstance(start, str): start = float(start)
        if isinstance(stop, str): stop= int(float(stop)) 
       
        this.log.debug('writing to: ' + str(out_fname))
        this.log.debug('start: ' + str(start))

        ffmpeg_args = ['ffmpeg']
        ffmpeg_args += ['-y']
        ffmpeg_args += ['-i', in_fname ]
        ffmpeg_args += ['-ss', str(start)]

        if stop:
            this.log.debug('duration: ' + str(stop-start))
            ffmpeg_args += ['-t', str(stop - start)]
        ffmpeg_args += [ out_fname ]
        
        this.log.debug('calling: ' + ' '.join(ffmpeg_args))
        process = subprocess.run(ffmpeg_args)
        if process.returncode != 0:
            this.log.ERR("Error: {} encountered by {}".format(
            process.returncode, clip_filename))
            return
 
        return        



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
    
    metas = [{  'YTID':'-2PDE7hUArE', 
                'start_seconds':'30.000', 
                'end_seconds': '40.000'}, 
             {  'YTID':'-DNkAalo7og', 
                'start_seconds':'30.000', 
                'end_seconds': '40.000'}, 
             {  'YTID':'-GDC7PuqdOM', 
                'start_seconds':'30.000', 
                'end_seconds': '40.000'}, 
             {  'YTID':'-jBWkHhQNew', 
                'start_seconds':'30.000', 
                'end_seconds': '40.000'}, 
             {  'YTID':'-x2aAKUtNRw', 
                'start_seconds':'30.000', 
                'end_seconds': '40.000'}] 
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
    clips_ytids = [ x['YTID'] for x in clips ]

    clips2 = dg.searchByYTIDs(clips_ytids)
    clips2_ytids = [ x['YTID'] for x in clips2 ]

    for x,y in zip(clips_ytids, clips2_ytids):
        assert( x==y)
    
    #these used to get reversed in early version of searchByYTIDs
    reverse_ytids = ['--S5Qr9ABZU', '--4qMR9M6tQ'] 
    clips3 = dg.searchByYTIDs(reverse_ytids)
    clips3_ytids = [ x['YTID'] for x in clips3 ]
    for x,y in zip(reverse_ytids, clips3_ytids):
        assert( x==y)


    dg = ClipFinder( audioset='eval,balanced,unbalanced', logLvl=logging.DEBUG)
    clip = dg.searchByYTIDs('zfLqqw47CrM')

    assert( clip[0]['YTID'] == 'zfLqqw47CrM')
    print (clip)

    try:
        dg = ClipFinder( audioset='balanaced,unbalanced', logLvl=logging.DEBUG)
        raise
    except: pass

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

    asd = AudioSetBuilder( data_dir=testdir, logLvl=logging.DEBUG,
                            audioset='balanced,unbalanced')
    ref_clips2 = [ '/home/mlbase/audioset/%s/--CIar_Kl4Y.wav' % testdir,
                    '/home/mlbase/audioset/%s/--jc0NAxK8M.wav' % testdir]
    metas = asd.info(ref_clips2)                    
    ytids = list(map(lambda x: x['YTID'], metas))
    assert ( ytids == ['--CIar_Kl4Y','--jc0NAxK8M']) 

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

    print ('TESTING info')

    metas = asd.info(ref_clips)
    meta_labels = list(map(lambda x: x['positive_labels'],metas))
    ref_labels = [ '/m/02mk9,/m/07pb8fc,/t/dd00066', '/m/02mk9,/m/07pb8fc,/t/dd00066']
    assert( all( x==y for x,y in zip( meta_labels, ref_labels)))

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
    #TestAudioSetBuilder( delete=False)

    print ('TESTING PASSED')
