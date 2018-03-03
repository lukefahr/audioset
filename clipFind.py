#!/usr/local/bin/python3

import csv
import logging
import itertools
import json
import multiprocessing
import os
import pydub 
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
class ClipFinder:

    class BreakException(Exception): pass
   
    def __init__(this, audioset='balanced,unbalanced', ontology=None, \
                    logLvl=logging.INFO):
        
        this.log = logging.getLogger( type(this).__name__)
        this.log.setLevel(logLvl)
       
        this.mydir = os.path.dirname(os.path.realpath(__file__))
     
        audioset = audioset.lower().split(',')
        this.audiosets = []
        if 'eval' in audioset:
            this.audiosets += [ this.mydir + '/eval_segments.csv'] 
        if 'balanced' in audioset:
            this.audiosets += [ this.mydir + '/balanced_train_segments.csv'] 
        if 'unbalanced' in audioset:
            this.audiosets += [this.mydir + '/unbalanced_train_segments.csv']
        
        this.ontol = ontology
        if this.ontol == None: this.ontol = this.mydir + '/ontology/ontology.json'
        this.log.info('opening ontology json: ' + str(this.ontol))
        this.ontol = open(this.ontol).read()
        this.ontol = json.loads(this.ontol)

        for audset in this.audiosets:
            assert( os.path.exists(audset) )

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

    def searchByYTIDs(this, ytids):
        ''' return a (youtubeID, start_time, end_time) tuple for the 
            given set of ytids
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

    def _searchByLabels( this, includes=[], excludes=[], max_clips=1):

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

                    if add == False:
                        # try to include first
                        for label in includes:
                            if label in row['positive_labels']: 
                                this.log.debug('Adding ' + row['YTID'] + ' (' + label + ')')
                                add= True
                                break

                    if add == True:
                        # try to exclude second
                        for label in excludes:
                            if label in row['positive_labels']: 
                                this.log.debug('Removing ' + row['YTID'] + ' (' + label + ')')
                                add= False
                                break

                    if add == True:
                        #this.log.debug('Actually adding ' + row['YTID'] ) 
                        clips.append( (row['YTID'],row['start_seconds'],
                                        row['end_seconds'] ) )

                    if max_clips!=None and len(clips) >= max_clips:
                        this.log.debug('Found sufficient number of clips found (' 
                                        + str(len(clips)) + '), stopping')
                        raise this.BreakException            
                
        except this.BreakException: pass

        this.log.debug('Total Clips : ' + str(len(clips)))

        return clips                        


    def _getIDbyNames(this, names):
        ''' selects ontology ID's by their corresponding human readable ontology names'''
        
        this.log.debug('Selecting Ontology ID by NAME(S): ' + str(names))

        if isinstance(names, str): names = [ names ]

        subs = [ x for x in this.ontol if x['name'] in names ]
        
        return [ x['id'] for x in subs ]

    def _openAudSet(this, fname):
        
        audset = open(fname, 'r')

        # the first few lines are not relivant
        this.log.debug( 'skipping: ' + str(audset.readline() ))
        this.log.debug( 'skipping: ' + str(audset.readline() ))
        this.log.debug( 'skipping: ' + str(audset.readline() ))

        audset= list(csv.DictReader(audset.readlines(), 
                        fieldnames=[ 'YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                        delimiter=',',
                        quotechar='"',
                        quoting = csv.QUOTE_ALL,
                        skipinitialspace=True))
        
        return audset

#
#
#
#
#
class GoodClipFinder( ClipFinder):
    
    def __init__(this, audioset='balanced,unbalanced', ontology=None, \
                    logLvl=logging.INFO):
                     
        super().__init__(audioset, ontology, logLvl)

        this.includes = ['Truck', 'Medium engine (mid frequency)', ], 
        this.excludes =[    'Air brake', 
                            'Air horn, truck horn', 
                            'Reversing beeps', 
                            'Ice cream truck, ice cream van', 
                            'Fire engine, fire truck (siren)',
                            'Jet engine', 
                           ] 

    def search( this, max_clips=1):
        return super().search( this.includes, this.excludes, max_clips)

#
#
#
#
#
class BadClipFinder( GoodClipFinder):
    
    def __init__(this, audioset='balanced,unbalanced', ontology=None, \
                    logLvl=logging.INFO):
                     
        super().__init__(audioset, ontology, logLvl)

        this.includes = ['all']
        this.excludes =[ 'Truck', 'Medium engine (mid frequency)', ] 

#
#
#
#
#
if __name__ == '__main__':
    
    logging.basicConfig ( level = logging.WARN,
                        format='%(levelname)s [0x%(process)x] %(name)s: %(message)s')
    
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

    gcf = GoodClipFinder( logLvl = logging.DEBUG)
    clips = gcf.search(2)    
    print (clips)
    assert( clips[0] == ('--PJHxphWEs', '30.000', '40.000'))

    bcf = BadClipFinder( logLvl = logging.DEBUG)
    clips = bcf.search(2)
    print (clips)
    assert( clips[0] ==('--PJHxphWEs', '30.000', '40.000') ) 





