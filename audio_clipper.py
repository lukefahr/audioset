#!/usr/bin/env python3
import wave
import os
import logging
import audioset
#
#
#
#
#
class AudioClipper(object):
    
    """
    Audio clipper class used to clip audio into chunks specified in milliseconds
    """
    
    def __init__ (this, chunk_duration, logLvl = logging.DEBUG):
        this.chunk_duration = chunk_duration ## in milliseconds
        this.chunk_dir = '{}_msec_chunks'.format(chunk_duration)
        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)


    def _get_output_dir(this, fname):
        split = os.path.split(os.path.abspath(fname))
        return split[0]+'/{}'.format(this.chunk_dir)
    
    
    def _get_chunk_abspath( this, fname, chunk_idx):
        split = os.path.split(os.path.abspath(fname))
        file_name, file_ext = os.path.splitext(split[1]) # file_name --PJHxphWEs, file_ext .wav
        op_dir = this._get_output_dir(fname)
        return os.path.join( op_dir, file_name+'_{}_'.format(chunk_idx)+file_ext)

    def _write_chunk_data(this, fname , data, params):
        r = -1
        try:
            wavf = wave.open(fname, 'wb')
            # (nchannels, sampwidth, framerate, nframes, comptype, compname)
            wavf.setparams( params)
            r = wavf.writeframes(data)
            wavf.close()
        except Exception as e:
            this.log.error('Wave: {}, write err: status {}'.format(fname, r))
        finally:
            pass
        return r

    def _chunk_file_exists(this, fname):
        return os.path.exists(fname)
    
    def chunk_file( this, fname):
        
        wavf = wave.open(fname, 'rb')
        frame_rate = wavf.getframerate()
        nframes = wavf.getnframes()
        nChannels = wavf.getnchannels()
        aud_params = wavf.getparams()
        ys = wavf.readframes(nframes)
        wavf.close()
        
        op_dir = this._get_output_dir(fname)
        chunks = []
        nSamples = round(this.chunk_duration*frame_rate/1000)

        this.log.debug( 'chunk_duration {} frame_rate {} nSamples {}'.format(this.chunk_duration, frame_rate, nSamples))
        
        if not os.path.exists( op_dir):
            try:
                this.log.debug('Creating directory path {}'.format(op_dir))
                os.makedirs(op_dir)
            except OSError as e:
                this.log.error('Failed to create directory {}'.format(op_dir))
                return chunks
        
        for i in range(0, len(ys), nSamples):
            start = i
            if i + nSamples > len(ys):
                stop = len(ys)
            else:
                stop = start + nSamples
            chunk_path = this._get_chunk_abspath(fname, i//nSamples)
            this.log.debug('chunk # {} start idx {} end idx {} >> n_samples {} path {}'.format(i//nSamples, start, stop, stop - start, chunk_path))

            # (nchannels, sampwidth, framerate, nframes, comptype, compname)
            params = list(aud_params)
            params[3] = params[1]//params[2] # num_frames

            r = this._write_chunk_data(chunk_path , ys[start:stop], params)
            if r != -1:
                chunks.append(chunk_path)

        this.log.info('Written chunks:' + '\n'.join([ c for c in chunks]))
        return chunks
                
            
class TestAudioClipper(object):

    def __init__(this):
        this.testdir = os.getcwd() + '/__testAudioClipper' 
        this.fname = this._download_clip(this.testdir)

    def _get_frame_rate( this, fname):
        ret = -1
        with wave.open(fname, 'rb') as wavf:
            ret = wavf.getframerate()
        return ret

    def _download_clip(this, testdir):
        asd = audioset.AudioSetBuilder( data_dir=testdir, logLvl=logging.DEBUG)
        clips = asd.getClips( includes= ['Truck', 'Medium engine (mid frequency)', ], 
                              excludes=[ 'Air brake', 
                                         'Air horn, truck horn', 
                                         'Reversing beeps', 
                                         'Ice cream truck, ice cream van', 
                                         'Fire engine, fire truck (siren)',
                                         'Jet engine', 
                              ], 
                              num_clips = 1, download=True, max_threads=1) 
        # print('Downloaded clip :', clips[0])
        this.original_audio_duration_ms = this._get_audio_duration_ms( clips[0])
        return clips[0]
    
    def _get_audio_duration_ms(this, fname):
        ret = -1
        with wave.open(fname, 'rb') as wavf:
            frameRate = wavf.getframerate()
            ys = wavf.readframes(wavf.getnframes())
            ret = len(ys)/frameRate*1000
        return ret

    def test(this, chunk_duration):
        clips = AudioClipper(chunk_duration, logLvl=logging.DEBUG).chunk_file(this.fname)
        # print('TEST clip {} total duration {} chunk duration {} # of clips {}'.\
              # format(this.fname,this.original_audio_duration_ms ,chunk_duration, len(clips)))
        c_dur = [ this._get_audio_duration_ms(c)  for c in clips]

        # print('# of chunks {}, sum of chunk durations {} '.format(len(clips), sum(c_dur)))
        assert( this.original_audio_duration_ms == sum(c_dur))

    
        
if __name__ == '__main__':
    test_chunk_lens = [1000, 2000, 3000, 100, 500 , 200]
    tester = TestAudioClipper()
    for c in test_chunk_lens:
        tester.test(c)
        
    print('TEST PASSED')
