#!/usr/bin/env python3

#the big import 
import glob
import json
from keras import backend as K #FIXME: K?
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (
                            Activation,
                            BatchNormalization,
                            concatenate,
                            Convolution2D,
                            Dense, Dropout,
                            Flatten,
                            GlobalAveragePooling2D, 
                            GlobalMaxPool2D, 
                            Input, 
                            MaxPool2D, 
                          )
from keras.utils import Sequence, to_categorical
import librosa
import logging
import numpy as np
import os
import time
import pandas as pd
import scipy
try:
    from sklearn.cross_validation import StratifiedKFold
except:
    from sklearn.model_selection import StratifiedKFold
import shutil
import sys
import wave

# and add the audioset path
MY_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append( MY_DIR + '/../audioset') 
import audioset

logging.basicConfig ( level = logging.WARN,
                    format='%(levelname)s %(name)s: %(message)s')

class AudioSplitter:

    class Config(dict):
        def __getattr__(self,attr):
            return self[attr]
        def __setattr__(self,attr,value):
            self[attr]=value

    #
    def __init__(this, data_dir=None, useDummyModel=False, sampling_rate = 16000, 
                max_threads = 5, num_epochs = 50, logLvl= logging.INFO):

        

        # our config datastructure 
        this.config = this.Config(  
                        # everything is based off this directory
                        data_dir = data_dir,

                        full_clip_dir = data_dir + '/full/',
                        part_clip_dir = data_dir + '/part/', 
                        csv_dir = data_dir + '/csv/',
                        models_dir = data_dir + '/models/',
                        self_prediction_dir = data_dir + '/models/prediction/', 
                        logs_dir = data_dir + '/logs',

                        full_clips_csv = data_dir + '/csv/full_clips.csv', 
                        part_clips_list_csv = data_dir + '/csv/partial_clips.all.clips.csv',
                        part_clips_classification_csv = data_dir + '/csv/partial_clips.all.classification.csv',
                        part_clips_full_output_csv = data_dir + '/csv/partial_clips.all.dataset.csv',
                        part_clips_output_csv = data_dir + '/csv/partial_clips.final.dataset.csv',
                        column_names_csv = data_dir + '/csv/model_column_names.csv',
                        
                        audioset = 'balanced,unbalanced', # audioset datasets to search
                        sampling_rate = sampling_rate,  #samples/second
                        full_clip_duration_sec= 10, # seconds 
                        num_classes= 2, # number of classes
                        num_folds= 2, # number of folds for training
                        learning_rate= 0.001, # DNN learning rate
                        max_epochs=  num_epochs, 
                        num_mfcc= 40, # number of MFCC coefficients

                        useDummyModel = useDummyModel, # use a simpler model (testing)
                        logLvl = logLvl,  # how chatty should we be
                        max_threads = max_threads, # max threads audioset can use 
                        ) 

        this.config.total_length = this.config.sampling_rate * this.config.full_clip_duration_sec
        this.config.dim = (  this.config.num_mfcc,
                                1 + int(np.floor(this.config.total_length/512)), 
                                1) # FIXME:  512?
        
        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
        this.log.info ("Created")

        try: os.makedirs(this.config.self_prediction_dir)
        except FileExistsError: pass
        try: os.makedirs(this.config.models_dir)
        except FileExistsError: pass
        try: os.makedirs(this.config.csv_dir)
        except FileExistsError: pass
        try: shutil.rmtree(this.config.logs_dir)
        except FileNotFoundError: pass
        
        # stash our config, just in case
        cfg_file = this.config.data_dir + '/config.json'
        this.log.debug('Saving config to: %s' % cfg_file) 
        with open(cfg_file, 'w') as fp:
            json.dump(this.config, fp)

    #
    def Run(this, labels, includes, excludes, num_clips, clip_length_ms):
        """
            Runs the AudioSplitter Algorithm, and returns a list of clips
            of length 'clip_length' that have been classified for 
            each label in 'labels'.  

            @labels:  the list of labels for the classifier
            @includes: the list of Ontology labels to be included under each label
                        empty lists are not allowed.  
            @excludes: the list on Ontology labels to be excluded under each label
                        empty lists ([]) are allowed.  
            @num_clips:  the maximum number of full clips to be included for each label
            @clip_length_ms:  the length (in milliseconds) of each classified partial clip

            @return:  a Pandas Dataframe of classified partial clips
        """
        cfg = this.config 
        
        this.log.debug('Running AudioSplitter')

        # download the clips
        if not os.path.isfile(this.config.full_clips_csv):
            this.log.info('Building full clips file...')
            this.GetFullClips(labels, includes, excludes, num_clips) 
        else:
            this.log.info('Full clips file already exists!')

        # Build the partial clips
        if not os.path.isfile(cfg.part_clips_list_csv):
            this.log.info('Building partial clips file...')
            this.BuildPartialClips( part_clip_time_ms=500)
        else:
            this.log.info('Partial clips file already exists!')

        # build model
        # (first check if any models exist)
        if len(glob.glob(cfg.models_dir + '/best*.h5')) == 0:
            this.log.info('Buildling Model...')
            this.BuildModel()
        else:
            this.log.info('Model already exists!')
       
        # classify the partial clips
        if not os.path.isfile( cfg.part_clips_classification_csv ):
            this.log.info('Classifying partial clips...')
            this.ClassifyClips()
        else:
            this.log.info('Partial Clips classification file already exists!')

        # build final dataset
        if not os.path.isfile( cfg.part_clips_output_csv):
            this.log.info('Building final dataset...')
            this.BuildFinalDataSet()
        else:
            this.log.info('Final dataset already exists!')
        
        out = pd.read_csv(cfg.part_clips_output_csv, index_col=0)
        return out

    #
    def GetFullClips(this, labels, includes, excludes, num_clips): 
        """
            Uses the AudioSetBuilder class to collect the full-length 
            audio clips from youtube. 

            @labels:  list of labels
            @includes:  list of Ontology values to include for each label
            @excludes:  list of ontology values to exclude for each label
            @num_clips:  the maximum number of clips for each label
        """
        cfg = this.config
        aset = audioset.AudioSetBuilder( audioset=cfg.audioset,
                                            sampling_rate = cfg.sampling_rate, 
                                            data_dir = cfg.full_clip_dir,
                                            logLvl = cfg.logLvl)

        clipFiles = pd.DataFrame() 
        
        for label, include, exclude in zip(labels, includes, excludes):

            this.log.debug('Working on label: %s' % label) 

            if include is None: include = []

            clipFiles = clipFiles.append( pd.DataFrame( {
                                    'label': label, 
                                    'file':  aset.getClips(
                                                label = label, 
                                                includes = include,
                                                excludes = exclude,
                                                num_clips = num_clips, 
                                                download = True, 
                                                max_threads = this.config.max_threads
                                              )
                                    } ), 
                                    ignore_index=True)

        clipFiles.to_csv( this.config.full_clips_csv, index=True, index_label = 'idx') 
    
    def BuildPartialClips(this, part_clip_time_ms = 500, 
                                full_clips_csv=None, part_clips_csv=None):
        """
            Builds a dataset of (unclassified) part clip files

            @ part_clip_time_ms: (optional) the length (in milliseconds) of each partial clip
            @full_clips_csv:  (optional) the input CSV file containing the full clips
            @part_clips_csv:  (optional) the output CSV file containing the partial clips

        """
        cfg = this.config

        if not full_clips_csv:
            full_clips_csv = cfg.full_clips_csv
        if not part_clips_csv:
            part_clips_csv = cfg.part_clips_list_csv

        this.log.debug('Building part clips:\n\tfrom:%s\n\tto:%s'%
                            (full_clips_csv,part_clips_csv))

        fullClips = pd.read_csv(full_clips_csv)
        partClips = pd.DataFrame() 

        if not os.path.exists(cfg.part_clip_dir):
            os.makedirs(cfg.part_clip_dir)
        
        for i in fullClips.index: 
            fullClip = fullClips.ix[i] 
            splitClips = this._splitClip(fullClip['file'], part_clip_time_ms,
                                    cfg.part_clip_dir)
            splitClips = pd.DataFrame( {'label': fullClip['label'], 'full_clip' : fullClip['file'], 
                        'part_clip' : splitClips})
            partClips = partClips.append(splitClips, ignore_index=True)                        

        partClips.to_csv( part_clips_csv, index=True, index_label='idx')
   
    #
    def BuildModel( this, clip_csv= None ):
        """
            Builds the DNN model used to classify partial clips

            @TODO:  document me
        """
        cfg = this.config

        if not clip_csv:
            clip_csv = cfg.full_clips_csv

        #first: load the clip lists
        clipFiles = pd.read_csv( clip_csv)

        #second : load the actual clip data
        this.log.debug('loading audio data')
        X_train = this._prepare_data(clipFiles)

        #third, index and binarize the labels
        this.log.debug('binarizing labels')
        y_train = pd.get_dummies(clipFiles['label'])

        # now we can actually build the model
        if (cfg.useDummyModel):
            model = this._buildDummyModel() 
        else: 
            model = this._buildModel() 

        # and run it 
        clipFiles['label_idx'] =  clipFiles['label'].astype('category').cat.codes
        try:
            skf = StratifiedKFold(clipFiles.label_idx, n_folds=cfg.num_folds)
        except TypeError:
            n_samples = len(clipFiles.label_idx)
            skf = StratifiedKFold(n_splits = cfg.num_folds)
            skf = skf.split( np.zeros(n_samples), clipFiles.label_idx) 

        for i, (train_split, val_split) in enumerate(skf):
            X, y = X_train[train_split],  y_train.values[train_split]
            X_val, y_val = X_train[val_split], y_train.values[val_split]

            checkpoint = ModelCheckpoint( cfg.models_dir + '/best_%d.h5'%i, monitor='val_loss', 
                                            verbose=1, save_best_only=True)
            early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        
            tb = TensorBoard(log_dir=cfg.logs_dir + '/fold_%i'%i, 
                            write_graph=True)
            
            callbacks_list = [checkpoint, early, tb]

            print("#"*50)
            print("Fold: ", i)

            history = model.fit(X, y, validation_data=(X_val, y_val), 
                        callbacks=callbacks_list, batch_size=64, epochs=cfg.max_epochs)

            # run predict on our test set                        
            model.load_weights(cfg.models_dir +'/best_%d.h5'%i)
            predictions = model.predict(X_train, batch_size=64, verbose=1)
           
            # save the column names for the model
            columns = pd.Series( ['name'] + list(y_train))
            columns.to_csv(cfg.column_names_csv)

            # Save train predictions
            np.save(cfg.self_prediction_dir + "/train_predictions_%d.npy"%i, predictions)
            y_predict =  pd.DataFrame(predictions, columns = list(y_train))
            y_predict.to_csv( cfg.self_prediction_dir + "/train_predictions_%d.csv"%i,
                                    index=True, index_label='idx')
    
    #
    def ClassifyClips(this, clipsFile=None, fileLabel='part_clip', outputFile=None):
        """
            Classify a series of (partial) clips

            @clipFile:
            @fileLabel:
            @outputFile:

        """
        cfg = this.config

        if not clipsFile:
            clipsFile = cfg.part_clips_list_csv
        if not outputFile:
            outputFile = cfg.part_clips_classification_csv

        clips = pd.read_csv(clipsFile)

        #second : load the actual clip data
        this.log.debug('loading test data')
        X_test = this._prepare_data(clips, fileLabel) 

        # third, load the model
        if (cfg.useDummyModel): model = this._buildDummyModel() 
        else: model = this._buildModel() 

        #fifth: load in the weights
        all_models = glob.glob(cfg.models_dir + '/best*.h5') 
        highest_model = '/best_' + str(max(map( lambda x: x.rsplit('.')[0].split('_')[-1], all_models))) + '.h5'
        model.load_weights(cfg.models_dir + highest_model)

        # now predict
        y_test = model.predict(X_test, batch_size=64, verbose=1)

        # load column names
        columnNames = pd.read_csv( cfg.column_names_csv)['name']
        
        # and save the results
        y_test = pd.DataFrame(y_test, columns = columnNames)
        y_test.to_csv( outputFile, index=True, index_label = 'idx')

    #
    def BuildFinalDataSet(this, partClips=None, partClassed=None, 
                                outFile=None):
        """
            Builds a dataset of partial clips that have also been classified 
            the same as their parent clip
            
            @partClips: (optional) the dataset of all partial clips
            @partClassed:  (optional) a dataset of all partial clips classifications
            @outFile:  (optional) the output dataset of the correctly-classified partial clips
        """
        this.log.debug('Building Final DataSet')
        cfg = this.config

        if not partClips: 
            partClips = pd.read_csv(cfg.part_clips_list_csv, index_col=0)
        if not partClassed:
            partClassed = pd.read_csv(cfg.part_clips_classification_csv, index_col=0)
        if not outFile:
            outFile = cfg.part_clips_output_csv
        
        this.log.debug('examining %d partial clips' % partClips.shape[0] )

        allClips = pd.concat([ partClips, partClassed], axis=1)
        allClips['selected'] = 0
        goodClips = pd.DataFrame()
        
        # loop over all classification labels, 
        # (skip the 'idx' column)
        for name in partClassed.columns: 
            this.log.debug('scanning for %s' % name)
            
            # select part_clip + label columns if the label matches and the 
            # partial-clip classification is correct (>50%)
            nameClips = allClips[ ['part_clip','label'] ] \
                        [(allClips[name] > 0.5) & (allClips['label'] == name)]
            goodClips = goodClips.append( nameClips, ignore_index = True)

            # note that we included it in the final output
            allClips.loc[ (allClips[name] > 0.5) & (allClips['label'] == name), 'selected'] = 1
        
        #rename the 'part_clip' column to 'file_name':
        goodClips.columns = list(map(lambda x: 'file_name' if x == 'part_clip' 
                                                        else x, goodClips.columns))

        allClips.to_csv(cfg.part_clips_full_output_csv, index=True, index_label='idx') 
        goodClips.to_csv(outFile, index=True, index_label='idx')




    #
    def _splitClip(this, clipfile, split_ms, output_folder):
        """
            Splits a clip into multiple smaller clips

            @clipfile:  the file name of the clip to be split
            @split_ms:  the time (in milliseconds) for each split clip
            @output_folder:  where should the split clips be stored

            @return:  a list of split file names
        """
        this.log.debug('splitting: %s' % clipfile)

        sr, ys = scipy.io.wavfile.read(clipfile)

        filename = os.path.splitext( os.path.basename(clipfile))[0]
        out_file_base = os.path.abspath(output_folder + '/' + filename)
        chunkFileNames = []
        
        T = int(sr * split_ms / 1000)
        chkIdx = 0
        for i in range(0, len(ys), T):
            outChunkFile = out_file_base + '.chk%i'%chkIdx + '.wav'
            scipy.io.wavfile.write( outChunkFile, data = ys[i:i+T], rate=sr)
            chunkFileNames.append(outChunkFile)
            chkIdx += 1

        return chunkFileNames

                           
    #        
    def _prepare_data(this, clips, file_label = 'file'):
        """
            Loads and pre-processes the data for Model building

            clips:  a dataframe of {'label','filename'} rows to be loaded
            file_label:  the column indicating the filename

        """

        config = this.config
        X = np.empty(shape=(clips.shape[0], config.dim[0], config.dim[1], 1))
        input_length = config.total_length
        for i,fname in enumerate(clips[file_label]):
            this.log.debug('loading %s' % fname)
            data, _ = librosa.core.load(fname, sr=config.sampling_rate, res_type="kaiser_fast")

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

            data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.num_mfcc)
            data = np.expand_dims(data, axis=-1)
            X[i,] = data

        # now normalize the data
        mean = np.mean(X, axis=0)
        stdev = np.std(X, axis=0)
        X = (X - mean) / stdev

        return X

    #
    def _buildDummyModel(this):
        """
            Returns a simple (dummy) NN model 
            Useful for testing
        """

        config = this.config
        nclass = config.num_classes
        inp = Input(shape=(config.dim[0],config.dim[1],1))
        x = Flatten()(inp)
        #x = Dense(64)(x)
        out = Dense(nclass, activation=softmax)(x)
        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)
        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    #
    def _buildModel(this):
        """ 
            Returns a more sophisticated DNN model
        """
        config = this.config
        nclass = config.num_classes

        inp = Input(shape=(config.dim[0],config.dim[1],1))
        x = Convolution2D(32, (4,10), padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Convolution2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Convolution2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Convolution2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model


# Testing
def UnitTest():
    """
        Unit Testing Function
    """

    print ("="*50)
    print (' '* 15 + 'Running Unit Testing')
    print ("="*50)
    time.sleep(0.5)

    test_dir = MY_DIR + '/__test'
    asp = AudioSplitter( data_dir = test_dir, useDummyModel=True, 
                sampling_rate = 8000, max_threads = 1, 
                num_epochs = 100,
                logLvl = logging.DEBUG)
    
    truck = {
        'label': 'truck',
        'includes': [ 'Truck', 'Medium engine (mid frequency)', ], 
        'excludes': [ 'Air brake', 'Air horn, truck horn', ],
        }
    notruck = {
        'label': 'notruck',
        'includes' : [],
        'excludes' : ['Truck', 'Medium engine (mid frequency)', ],
        }

    builder = (truck, notruck)
    labels = list(map(lambda x: x['label'], builder))
    includes = list(map(lambda x: x['includes'], builder))
    excludes = list(map(lambda x: x['excludes'], builder))
    num_clips = 10 

    data = asp.Run(labels, includes, excludes, num_clips = num_clips, clip_length_ms=1000) 
    #asp.GetFullClips(labels, includes, excludes, num_clips = num_clips) 
    #asp.BuildModel()
    #asp.BuildPartialClips( part_clip_time_ms=1000)
    #asp.ClassifyClips()
    #asp.BuildFinalDataSet()

    print ("="*50)
    print (' '* 15 + 'Passed Unit Testing')
    print ("="*50)
    time.sleep(0.5)

#
#
#
#
#
if __name__ == '__main__':

    logging.basicConfig ( level = logging.WARN,
                        format='%(levelname)s [0x%(process)x] %(name)s: %(message)s')
    if (len(sys.argv) > 1) and ( sys.argv[1] == '--unit'):
        UnitTest()
    else:
        print ("Please specify the a '--unit' argument for unit testing")

