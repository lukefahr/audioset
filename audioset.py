
import copy
import logging
import numpy as np
import os
import random
import pydub

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
    
    class iter_wave(object):
        def __init__(this, obj, ytids, subdir):
            this.obj = obj
            this.iter = iter(ytids)
            this.subdir = subdir
            this.ytid=None

        def __next__(this):
            ytid = next(this.iter)
            this.obj.log.debug ('ytid: ' + str(ytid) )
            return this.obj._getSampleData(ytid, this.subdir)
        

    def __init__(this, framerate=22000, rand_seed=42, 
                    data_dir=None, logLvl=logging.INFO):

        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
        
        this.rnd = random.Random(rand_seed)

        this.mydir = os.path.dirname(os.path.realpath(__file__))

        this.ddir = this.mydir+'/_data' if data_dir  == None \
                        else data_dir

        this.framerate = framerate

        assert( os.path.exists(this.ddir))
        assert( this.framerate > 0 and this.framerate < 1e6)


    def _getYTIDs(this, subdir, maximum=None):
        subdir = this.ddir + '/' + subdir + '/'
        assert( os.path.exists( subdir))

        this.log.debug('Collecting YTIDs in ' + str(subdir))

        all_files = [name for name in os.listdir(subdir) ]
        all_waves = [ name for name in all_files \
                            if '.wav' in name ]
        all_ytids = [ x.split('.wav')[0] for x in all_waves]
        
        if maximum != None and len(all_ytids) > maximum:
            this.log.debug('downsampling ytids to ' + str(maximum) )
            all_ytids = this.rnd.sample(all_ytids, maximum)

        return all_ytids

    def _getSampleData(this, ytid, subdir):
        
        this.log.info('Loading ytid: ' + str(ytid))

        fname = this.ddir + '/' + subdir + '/' + ytid + '.wav'
        assert( os.path.exists(fname))
        
        rate = str(int(this.framerate/1000)) + 'k'
        fname_rate = this.ddir + '/' + subdir + '/' + \
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


class GoodAudioSetData (AudioSetData):

    def __init__(this, num_samples=1, subdir='good', 
                    framerate=22000, rand_seed=42, 
                    data_dir=None, logLvl=logging.INFO):

        super().__init__(framerate, rand_seed, data_dir, logLvl)
        
        this.ytids = this._getYTIDs( subdir, num_samples)
        this.subdir = subdir

    def __iter__(this):
        return this.iter_wave(this, this.ytids, this.subdir)
   
    def __getitem__(this, key):
        ytid = this.ytids[key]
        this.log.debug( 'ytid: ' + str(ytid))
        return this._getSampleData(ytid, this.subdir)

class BadAudioSetData (GoodAudioSetData):

    def __init__(this, num_samples=1, subdir='bad', 
                    framerate=22000, rand_seed=42, 
                    data_dir=None, logLvl=logging.INFO):

        super().__init__(num_samples, subdir, 
                    framerate, rand_seed, 
                    data_dir, logLvl)
        
    
if __name__ == '__main__':

    logging.basicConfig ( level = logging.WARN,
                    format='%(levelname)s %(name)s: %(message)s')
    
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
    asd = AudioSetData(logLvl=logging.DEBUG)
    data = asd._getSampleData('zfLqqw47CrM', 'good')
    assert ( data.ys[0] == 2032)

    print ('TESTING')
    os.remove('_data/good/22k/zfLqqw47CrM.wav')
    data = asd._getSampleData('zfLqqw47CrM', 'good')
    assert ( data.ys[0] == 2032)

    print ('TESTING')
    ytids = asd._getYTIDs('good')
    assert ( ytids[0] == '1JgSGO2W7Bk')

    print ('TESTING')
    ytids = asd._getYTIDs('good',10)
    assert (ytids[0] == '-wyr5_KppiU' )

    print ('TESTING')
    gasd = GoodAudioSetData(2, logLvl=logging.DEBUG)
    ig = iter(gasd)
    assert( next(ig).ys[0] == 4165 )
    assert( next(ig).ys[0] == 1472 )
    assert( gasd[1].ys[0] == 1472 )

    basd = BadAudioSetData(2, logLvl = logging.DEBUG)
    ib = iter(basd)
    assert( next(ib).ys[0] == 81)

