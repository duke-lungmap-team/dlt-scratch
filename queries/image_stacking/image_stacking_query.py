# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 08:41:56 2016

@author: nn31
"""

import pandas as pd

#read in all of the metadata pulled from website
metadata   = pd.read_csv(open('/Users/nn31/Dropbox/40-githubRrepos/lungmap-data-plus-2016/metadata/00_metadata_dictionary.csv','r'))
researcher = pd.read_csv(open('/Users/nn31/Dropbox/40-githubRrepos/lungmap-data-plus-2016/metadata/01_metadata_researcher.csv','r'))
experiment = pd.read_csv(open('/Users/nn31/Dropbox/40-githubRrepos/lungmap-data-plus-2016/metadata/02_metadata_experiment_type.csv','r'))
probe      = pd.read_csv(open('/Users/nn31/Dropbox/40-githubRrepos/lungmap-data-plus-2016/metadata/03_metadata_probe_antibody.csv','r'))
sample     = pd.read_csv(open('/Users/nn31/Dropbox/40-githubRrepos/lungmap-data-plus-2016/metadata/04_metadata_sample_details.csv','r'))
annotation = pd.read_csv(open('/Users/nn31/Dropbox/40-githubRrepos/lungmap-data-plus-2016/metadata/05_metadata_experiment_annotation.csv','r'))
images     = pd.read_csv(open('/Users/nn31/Dropbox/40-githubRrepos/lungmap-data-plus-2016/metadata/06_metadata_experiment_images.csv','r'))


#Let's start with 1 sample (i.e. set of lungs)
sample_try = 'LMSP0000000032'
first_sample = sample[sample['sample_id']==sample_try]
#here are all the experiments from that set of lungs
print(first_sample['experiment_id'])
#get all images associated with a particular experiment
experiment_try = 'LMEX0000000264'
print(images[images['experiment']==experiment_try])
#now, i think the magnifications represent the "stack", let's check
mag = '100X'
images_20x = images[(images['experiment']==experiment_try) & (images['magnification']==mag)]['image'].tolist()
#now let's get the location on S3 for these images
import pprint
print('Checking Experiment: ' + experiment_try)
print('At magnification: ' + mag)
print('*******************')
pprint.pprint(metadata[metadata['image_id'].isin(images_20x)]['s3downloadkey'].tolist())