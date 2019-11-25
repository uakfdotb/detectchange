from discoverlib import geom
import model as model

import numpy
import os
from PIL import Image
import random
import scipy.ndimage
import subprocess
import sys
import tensorflow as tf
import time

SIZE = 256
PATH = '/home/ubuntu/2019aug29-satupdate-doha-pickone/'
SAT_PATHS = [
	'/data/discover-datasets/2018may10-ken/doha/sat-jpg-old/',
	'/data/discover-datasets/2018may10-ken/doha/sat-jpg/',
]
OSM_PATH = '/data/discover-datasets/2018may10-ken/doha/osmtile-12px/'
TRAIN_REGIONS = ['doha']

osm_keys = set([fname.split('_osm.png')[0] for fname in os.listdir(OSM_PATH) if '_osm.png' in fname])
sat_keys = set([fname.split('_sat.jpg')[0] for fname in os.listdir(SAT_PATHS[0]) if '_sat.jpg' in fname])
keys = osm_keys.intersection(sat_keys)
train_keys = [k for k in keys if k.split('_')[0] in TRAIN_REGIONS]

random.shuffle(train_keys)
num_val = len(train_keys)/10
val_keys = train_keys[:num_val]
train_keys = train_keys[num_val:]

def load_tile(sat_path, k):
	region, x, y = k.split('_')
	sat_fname = '{}/{}_{}_{}_sat.jpg'.format(sat_path, region, x, y)
	print 'load {}'.format(sat_fname)
	im = scipy.ndimage.imread(sat_fname)[:, :, 0:3]
	osm_fname = '{}/{}_{}_{}_osm.png'.format(OSM_PATH, region, x, y)
	osm = scipy.ndimage.imread(osm_fname)
	osm = scipy.ndimage.zoom(osm, 0.25, order=1)
	osm = numpy.expand_dims(osm, axis=2)

	return im.swapaxes(0, 1), osm.swapaxes(0, 1)

for sat_path in SAT_PATHS:
	print 'reading {} train tiles from {}'.format(len(train_keys), sat_path)
	train_tiles = [load_tile(sat_path, k) for k in train_keys]
	print 'reading {} val tiles from {}'.format(len(val_keys), sat_path)
	val_tiles = [load_tile(sat_path, k) for k in val_keys]

def extract(tiles):
	tile = random.choice(tiles)
	i = random.randint(0, (4096-SIZE)/4)
	j = random.randint(0, (4096-SIZE)/4)
	im = tile[0][i*4:(i*4)+SIZE, j*4:(j*4)+SIZE, :]
	osm = tile[1][i:i+SIZE/4, j:j+SIZE/4, :]
	return im, osm

val_rects = [extract(val_tiles) for _ in xrange(256)]

print 'initializing model'
m = model.Model(size=SIZE)
session = tf.Session()
session.run(m.init_op)

for s in ['model_latest', 'model_best']:
	try:
		os.mkdir('{}/{}'.format(PATH, s))
	except:
		pass

latest_path = '{}/model_latest/model'.format(PATH)
best_path = '{}/model_best/model'.format(PATH)

print 'begin training'
best_loss = None

for epoch in xrange(9999):
	start_time = time.time()
	train_losses = []
	for _ in xrange(1024):
		batch_tiles = [extract(train_tiles) for _ in xrange(model.BATCH_SIZE)]
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.is_training: True,
			m.inputs: [tile[0].astype('float32') / 255.0 for tile in batch_tiles],
			m.targets: [tile[1].astype('float32') / 255.0 for tile in batch_tiles],
			m.learning_rate: 1e-3,
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in xrange(0, len(val_rects), model.BATCH_SIZE):
		batch_tiles = val_rects[i:i+model.BATCH_SIZE]
		outputs, loss = session.run([m.outputs, m.loss], feed_dict={
			m.is_training: False,
			m.inputs: [tile[0].astype('float32') / 255.0 for tile in batch_tiles],
			m.targets: [tile[1].astype('float32') / 255.0 for tile in batch_tiles],
		})
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print 'iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss)

	m.saver.save(session, latest_path)
	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, best_path)

def test():
	for i, tile in enumerate(val_rects[0:64]):
		outputs, targets_det, outputs_det = session.run([m.outputs, m.targets_det, m.outputs_det], feed_dict={
			m.inputs: [tile[0].astype('float32') / 255.0],
			m.targets: [tile[1].astype('float32') / 255.0],
		})
		targets_det = numpy.clip(targets_det*80+0.5, 0, 1)
		outputs_det = numpy.clip(outputs_det*80+0.5, 0, 1)
		Image.fromarray((outputs[0, :, :, 0]*255).astype('uint8')).save('/home/ubuntu/vis/{}_out.png'.format(i))
		Image.fromarray((targets_det[0, :, :, 0]*255).astype('uint8')).save('/home/ubuntu/vis/{}_td.png'.format(i))
		Image.fromarray((outputs_det[0, :, :, 0]*255).astype('uint8')).save('/home/ubuntu/vis/{}_od.png'.format(i))
