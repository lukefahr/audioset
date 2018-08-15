
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
        
        this.ys = ys
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

    def normalize(this, amp=1.0):
        high, low = abs(max(this._ys)), abs(min(this._ys))
        new_ys = amp * this._ys / max(high, low)
        this.ys = new_ys
    
    def getNames(this,):
        try: return this._names
        except AttributeError:
            cf= clipFind.ClipFinder( audioset='eval,balanced,unbalanced')
            this._names = cf.nameLookup(this.ytid)
            return this._names

    @property
    def ys(this):
        return this._ys

    @ys.setter
    def ys(this,ys):
        this._ys = np.array(ys)

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

        this.ddir = this.mydir+'/_data' if data_dir  == None \
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
                                    logLvl=this.log.getEffectiveLevel())
        metas = cf.searchByYTIDs(ytids)

        return this._downloadMetas(metas, threads)
    
    def _downloadMetas(this, metas, threads=1):

        this.log.debug('starting download')
        cdl = clipDownload.ClipDownloader( data_dir = this.ddir , 
                    logLvl=this.log.getEffectiveLevel())
        cdl.download(metas, max_threads = threads)

    def _getMetasByYTIDs(this, ytids):
        this.log.debug('collecting metas for random ytids')
        cf = clipFind.ClipFinder( audioset='eval,balanced,unbalanced', 
                                    logLvl=this.log.getEffectiveLevel())
        return cf.searchByYTIDs(ytids)

    
    def _verifyMetas(this, metas):
        ret_metas = []

        for meta in metas:
            ytid= meta[0]
            
            if os.path.exists(this.ddir + '/' + ytid + '.wav'):
                ret_metas.append( meta)

        return ret_metas


class GoodAudioSetData (AudioSetData):
    
    class NotFoundException(Exception): pass

    class iter_wave(object):
        def __init__(this, obj, metas):
            this.obj = obj
            this.iter = iter(metas)

        def __next__(this):
            meta = next(this.iter)
            this.obj.log.debug ('meta: ' + str(meta) )
            return this.obj._getSampleData(meta[0], )
 
    def __init__(this, num_samples=1,  download=False,
                    framerate=22000, max_threads = 1,
                    data_dir=None, logLvl=logging.INFO):

        assert( num_samples >= 0)

        super().__init__(framerate, data_dir, logLvl)
       
        this.download = download
       
        this.metas = []
        
        # no metas, that's fine
        if num_samples == 0: return

        meta_num = num_samples
        for i in range(5,0,-1): # try at most 5 round of expanded searches

            this.log.debug('collecting ' + str(meta_num) + ' metadatas')
            raw_metas= this._getMetas( meta_num )

            if this.download:
                this.log.debug('downloading clips')
                this._downloadMetas(raw_metas, max_threads)
        
            this.metas = this._verifyMetas(raw_metas)

            if len(raw_metas) < meta_num:
                this.log.info ('insufficient raw metas available: ' + 
                            str(len(raw_metas)) )
                break
            elif len(this.metas) < num_samples:
                this.log.debug('some metas not downloaded')
                this.log.debug('have: ' + str(len(this.metas)) + 
                                    ' want: ' + str(num_samples))
                if i > 1: this.log.debug(' trying bigger search')
                else: this.log.debug(' giving up')
                meta_num *= 2
            else:
                this.log.debug('found sufficient metas')
                break

        this.metas = this.metas[:num_samples]
    
    @property
    def metasList(this):
        return this.metas

    def __len__(this):
        return len(this.metas)

    def __iter__(this):
        return this.iter_wave(this, this.metas)
   
    def __getitem__(this, key):
        if isinstance(key, int):
            ytid = this.metas[key][0]
        else: 
            if key not in [ m[0] for m in this.metas]:
                this.log.debug('adding meta: ' + str(key))
                this._addClips( [key] )
            ytid = key

        this.log.debug( '[ytid]: ' + str(ytid))
        return this._getSampleData(ytid )

    def _getMetas(this, maximum=None):
        
        this.log.debug('collecting metadatas for ' + str(maximum) + 'clips')
        cf = clipFind.GoodClipFinder( logLvl=this.log.getEffectiveLevel() )
        return cf.search(maximum)

    def _addClips( this, ytids):
            metas = this._getMetasByYTIDs(ytids)

            if this.download:
                this._downloadMetas(metas)
            
            if this._verifyMetas( metas)  == []:
                raise this.NotFoundException("Meta/s not found")

            this.metas += metas
            

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
    data = AudioData([0,0,5,0,0], 4, 'biteme')
    print ( data.ys)
    data.normalize()
    print ( list(map( lambda x,y: x==y, data.ys, [0,0,1.,0,0]) ))
    assert( sum( map(lambda x,y: x==y, data.ys , [0,0,1.,0,0])) == 5 )


    print ('TESTING')
    ytid = '2WI5dfkw4-k'
    asd = AudioSetData(data_dir = testdir, logLvl=logging.DEBUG)
    assert( os.path.exists( testdir + '/'+ ytid + '.wav') == False)
    asd._downloadYTIDs('2WI5dfkw4-k')
    assert( os.path.exists( testdir + '/'+ ytid + '.wav') ==  True)
    asd._downloadYTIDs('2WI5dfkw4-k')
    assert( os.path.exists( testdir + '/'+ ytid + '.wav') ==  True)

    print ('TESTING')
    asd = AudioSetData(data_dir = testdir, logLvl=logging.DEBUG)
    meta = asd._getMetasByYTIDs('BJb9Idgq_xo')
    print (meta)
    assert(meta[0] == ('BJb9Idgq_xo', '30.000', '40.000') )  

    print ('TESTING')
    ytid = '2WI5dfkw4-k'
    data = asd._getSampleData(ytid)
    print (data.ys[:10])
    assert ( data.ys[0] == 9433)
    data = asd._getSampleData(ytid)
    assert ( data.ys[0] == 9433)
   
    print ('TESTING')
    gasd = GoodAudioSetData(2, download=True, data_dir=testdir, logLvl=logging.DEBUG)
    data = gasd['BJb9Idgq_xo']
    assert( data.ytid == 'BJb9Idgq_xo')

    print ('TESTING')
    gasd = GoodAudioSetData(2, download=False, data_dir=testdir, logLvl=logging.DEBUG)
    print (gasd[0].ys[:10])
    print (gasd[0].ytid)
    assert( gasd[0].ys[0] == 1282)
    assert( gasd[0].ytid == '-2PDE7hUArE')

    gitr = iter(gasd)
    data = next(gitr)
    assert( data.ys[0] == 1282)
    assert( data.ytid == '-2PDE7hUArE')

    print ('TESTING')
    try:
        data = gasd['zfLqqw47CrM']
        raise
    except gasd.NotFoundException: pass

    print ('TESTING')
    gasd = GoodAudioSetData(2, download=True, data_dir=testdir, logLvl=logging.DEBUG)
    ig = iter(gasd)
    assert( next(ig).ytid == '-2PDE7hUArE')
    assert( next(ig).ytid == '-DNkAalo7og')

    print ('TESTING')
    basd = BadAudioSetData(2, download=True, data_dir=testdir, logLvl = logging.DEBUG)
    ib = iter(basd)
    assert( next(ib).ytid == '--PJHxphWEs')
    assert( next(ib).ytid == '--ZhevVpy1s' )

    print ('TESTING')
    try:
        print (next(ib).ys[0])
        raise
    except StopIteration:
        pass

    print ('TESTING')
    gasd = GoodAudioSetData(10, download=False, data_dir=testdir, logLvl=logging.DEBUG)
    ig = iter(gasd)
    print ( next(ig).ytid)
    print ( next(ig).ytid)
    try:
        print ( next(ig).ytid)
        raise
    except StopIteration:
        pass

    print ('TESTING COMPLETE!')
    
