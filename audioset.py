
import copy
import logging
import numpy as np
import os
import shutil
import random
import pydub

import clipFind
import clipDownload

try:
    from IPython.display import Audio
except: 
    pass

class AudioData(object):
    def __init__(this, ys, framerate, ytid):
        
        this._ys = ys
        this.framerate = framerate
        this.ytid = ytid

    def _fft( this, ):
        hs = np.fft.rfft( this._ys)
        return hs
    
    def make_audio(this):
        a = Audio(this._ys.real, rate=this.framerate)
        return a
    
    def copy(this):
        return copy.deepcopy(this)

    @property
    def ys(this):
        return this._ys

    @ys.setter
    def ys(this,ys):
        this._ys = ys

        try: del this._hs
        except AttributeError: pass
        try: del this._fs
        except AttributeError: pass

    @property
    def time(this):
        return len(this._ys) / this.framerate
    
    @property
    def hs(this):
        try:
            return this._hs
        except AttributeError:
            this._hs = np.fft.rfft(this._ys)
            return this._hs
   
    @property
    def fs(this):
        try:
            return this._fs
        except AttributeError:
            this._fs = np.fft.rfftfreq( len(this._ys), 1. / this.framerate)
            return this._fs


class AudioSetData (object):
    
       

    def __init__(this, framerate=22000, 
                    data_dir=None, logLvl=logging.INFO):

        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
        
        this.mydir = os.path.dirname(os.path.realpath(__file__))

        this.ddir = this.mydir+'/_unified_data' if data_dir  == None \
                        else data_dir

        this.framerate = framerate

        assert( this.framerate > 0 and this.framerate < 1e6)


    def _getSampleData(this, ytid):
        
        this.log.info('Loading ytid: ' + str(ytid))

        fname = this.ddir + '/' + ytid + '.wav'
        assert( os.path.exists(fname))
       
        rate = 'r_' + str(int(this.framerate)) 
        fname_rate = this.ddir + '/' + \
                rate + '/' + ytid + '.wav'
        if not os.path.exists(fname_rate):
            this._downsample(fname, this.framerate, fname_rate)
        
        aud = pydub.AudioSegment.from_wav( fname_rate)
        aud = aud.set_channels(1)
        data = aud.get_array_of_samples()
        data = np.asarray(data)

        return AudioData (data, this.framerate, ytid)


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


    def _downloadYTIDs(this, ytids, threads=1): 
        ''' download a list of ytids into the unified folder '''
        
        this.log.debug('downloading ytids')
        
        if isinstance(ytids, str): ytids = [ytids]

        exclude = []
        for ytid in ytids:
            #if it's already there, skip downloading
            if os.path.exists( this.ddir + '/' + str(ytid) + '.wav'):
                this.log.debug('ytid: ' + str(ytid) + ' already exists')
                exclude += [ytid]
        
        #this is not that efficient...
        ytids = [x for x in ytids if x not in exclude ]

        if len(ytids) == 0:
            this.log.debug('all metas already present')
            return

        this.log.debug('gathering metadatas for ytids')
        cf = clipFind.ClipFinder( audioset='eval,balanced,unbalanced', 
                                    logLvl=logging.DEBUG)
        metas = cf.searchByYTIDs(ytids)

        return this._downloadMetas(metas, threads)
    
    def _downloadMetas(this, metas, threads=1):

        this.log.debug('starting download')
        cdl = clipDownload.ClipDownloader( data_dir = this.ddir , 
                    logLvl=this.log.getEffectiveLevel())
        cdl.download(metas, max_threads = threads)




class GoodAudioSetData (AudioSetData):

    class iter_wave(object):
        def __init__(this, obj, metas):
            this.obj = obj
            this.iter = iter(metas)

        def __next__(this):
            meta = next(this.iter)
            this.obj.log.debug ('meta: ' + str(meta) )
            return this.obj._getSampleData(meta[0], )
 
    def __init__(this, num_samples=1, 
                    framerate=22000, max_threads = 1,
                    data_dir=None, logLvl=logging.INFO):

        super().__init__(framerate, data_dir, logLvl)
       
        this.log.debug('collecting clip metadatas')
        this.metas= this._getMetas( num_samples)
        
        this.log.debug('downloading clips')
        this._downloadMetas(this.metas, max_threads)

    def __iter__(this):
        return this.iter_wave(this, this.metas)
   
    def __getitem__(this, key):
        if isinstance(key, int):
            ytid = this.metas[key][0]
        else: 
            ytid = key
            assert( ytid in [ m[0] for m in this.metas])

        this.log.debug( '[ytid]: ' + str(ytid))
        return this._getSampleData(ytid )

    def _getMetas(this, maximum=None):

        this.log.debug('collecting metadatas for ' + str(maximum) + 'clips')
        cf = clipFind.GoodClipFinder( logLvl=this.log.getEffectiveLevel() )
        return cf.search(maximum)


class BadAudioSetData (GoodAudioSetData):

     def _getMetas(this, maximum=None):

        this.log.debug('collecting metadatas for ' + str(maximum) + 'clips')
        cf = clipFind.BadClipFinder( 
                                logLvl=this.log.getEffectiveLevel())
        return cf.search(maximum)

       
    
if __name__ == '__main__':

    logging.basicConfig ( level = logging.WARN,
                    format='%(levelname)s %(name)s: %(message)s')
    
    testdir = os.getcwd() + '/__test' 
    
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    
    print ('TESTING')
    data = AudioData([0,0,1,0,0], 4 , 'biteme')
    print ( data.ys )
    data.ys = [ 0,0,2,0,0]
    print ( data.ys)

    print ('TESTING')
    data = AudioData([0,0,1,0,0], 4, 'biteme')
    print ( data.hs)
    data.ys = [ 0,0,2,0,0]
    print ( data.hs)

    print ('TESTING')
    ytid = '2WI5dfkw4-k'
    asd = AudioSetData(data_dir = testdir, logLvl=logging.DEBUG)
    assert( os.path.exists( testdir + '/'+ ytid + '.wav') == False)
    asd._downloadYTIDs('2WI5dfkw4-k')
    assert( os.path.exists( testdir + '/'+ ytid + '.wav') ==  True)
    asd._downloadYTIDs('2WI5dfkw4-k')
    assert( os.path.exists( testdir + '/'+ ytid + '.wav') ==  True)

    print ('TESTING')
    data = asd._getSampleData(ytid)
    print (data.ys[:10])
    assert ( data.ys[0] == 9433)
    data = asd._getSampleData(ytid)
    assert ( data.ys[0] == 9433)
   
    print ('TESTING')
    gasd = GoodAudioSetData(2, data_dir=testdir, logLvl=logging.DEBUG)
    print (gasd[0].ys[:10])
    assert( gasd[0].ys[0] == 3160)
    assert( gasd[0].ytid == '--PJHxphWEs')

    gitr = iter(gasd)
    data = next(gitr)
    assert( data.ys[0] == 3160)
    assert( data.ytid == '--PJHxphWEs')

    print ('TESTING')
    try:
        data = gasd['zfLqqw47CrM']
        raise
    except AssertionError: pass

    print ('TESTING')
    gasd = GoodAudioSetData(2, data_dir=testdir, logLvl=logging.DEBUG)
    ig = iter(gasd)
    assert( next(ig).ytid == '--PJHxphWEs')
    assert( next(ig).ytid == '--ZhevVpy1s')

    print ('TESTING')
    basd = BadAudioSetData(2, logLvl = logging.DEBUG)
    ib = iter(basd)
    assert( next(ib).ytid == '--PJHxphWEs')
    assert( next(ib).ytid == '--ZhevVpy1s' )

    print ('TESTING')
    try:
        print (next(ib).ys[0])
        raise
    except StopIteration:
        pass

    print ('TESTING COMPLETE!')
    
