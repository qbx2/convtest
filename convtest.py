import tensorflow as tf
import gym
import pyglet

env = gym.make('Breakout-v0')
sess = tf.InteractiveSession()

# NHWC, 
def convolute(x, f, stride, padding):
	x = tf.expand_dims(x, 0)
	f = tf.expand_dims(tf.expand_dims(tf.cast(f, tf.float32), -1), -1)
	return tf.nn.conv2d(x, f, strides=[1, stride, stride, 1], padding=padding, data_format='NHWC').eval()[0]

def getX(ob): # (210, 160, 3)
	grayscale = tf.image.rgb_to_grayscale(ob) # (210, 160, 1)
	cropped = tf.image.crop_to_bounding_box(grayscale, 32, 8, 164, 144)
	#tf.image.resize_image_with_crop_or_pad(grayscale[32:32+164, 8:8+144, :], 164, 144)
	expanded = tf.expand_dims(cropped, 0) # (1, 210, 160, 1)
	resized = expanded#tf.image.resize_bicubic(expanded, (164//2, 144//2))
	resized = resized[0, :, :, :]
	padded = resized#tf.image.pad_to_bounding_box(resized, 4, 4, 164//2+4+4, 144//2+4+4)
	return (tf.cast(padded, tf.float32)/255.).eval()

def save(img, filename='test.png'):
	with open(filename, 'wb') as f:
		f.write(tf.image.encode_png(tf.cast(tf.clip_by_value(tf.cast(img, tf.float32), 0, 255), tf.uint8)).eval())

def test(ob, f, s, padding, filename='test.png'):
	img = convolute(getX(ob), f, s, padding)
	if s == 4:
		tf.Print(img, [img], summarize=10000).eval()
	save(img*255, filename)

ob = env.reset()

for i in range(20):
	ob,*_=env.step(1)
	env.render()

PADDING = 'VALID'

#import code
#code.interact(local=locals())

f = tf.zeros([7, 7]).eval()
f[3, 3] = 1
save(ob[32:32+164, 8:8+144, :], 'orig.png')
test(ob, f, 1, PADDING, '7x7_identity_stride_1.png')
test(ob, f, 2, PADDING, '7x7_identity_stride_2.png')
test(ob, f, 3, PADDING, '7x7_identity_stride_3.png')
test(ob, f, 4, PADDING, '7x7_identity_stride_4.png')
test(ob, f, 8, PADDING, '7x7_identity_stride_8.png')

import code
code.interact(local=locals())
