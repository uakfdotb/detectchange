from discoverlib import geom, graph, tf_util
import model

import json
import numpy
import math
import os
import os.path
from PIL import Image
import random
import scipy.ndimage
import sys
import tensorflow as tf
import time

MODEL_PATH = '/data/discover-models/2019feb02-mass-seg-pickone/model_best/model'

print 'initializing model'
m = model.Model(size=2048)
session = tf.Session()
m.saver.restore(session, MODEL_PATH)

dir_pairs = [
	(
		'/data/discover-datasets/2018may10-ken/doha/sat-jpg-old/',
		'/data/discover-datasets/2018may10-ken/doha/seg-old/',
	),
	(
		'/data/discover-datasets/2018may10-ken/doha/sat-jpg/',
		'/data/discover-datasets/2018may10-ken/doha/seg/',
	),
]

for sat_path, seg_path in dir_pairs:
	for fname in os.listdir(sat_path):
		if '_sat.jpg' not in fname:
			continue
		label = fname.split('_sat.jpg')[0]
		sat_fname = sat_path + label + "_sat.jpg"
		out_fname = seg_path + label + '_seg.png'
		print 'run', label
		im = scipy.ndimage.imread(sat_fname)[:, :, 0:3]
		out = tf_util.apply_conv(session, m, im, scale=4, channels=64)
		out = (numpy.max(out, axis=2) * 255).astype('uint8')
		Image.fromarray(out).save(out_fname)
