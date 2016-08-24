# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:08:25 2016

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

#Seems the only annotated images that we have access to are 20x, so that is what we'll start with
mag = '20x'
images_20x = images[(images['magnification']==mag)]['image'].tolist()