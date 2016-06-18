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
	cropped = tf.image.resize_image_with_crop_or_pad(grayscale[32:, :, :], 210, 144)
	resized = tf.image.resize_images(cropped, 89, 72)/255. # (89, 72, 1)
	return resized.eval()

def test(ob, f, s, padding, filename='test.png'):
	img = convolute(getX(ob), f, s, padding)

	with open(filename, 'wb') as f:
		f.write(tf.image.encode_png(tf.cast(tf.clip_by_value(img*255, 0, 255), tf.uint8)).eval())

ob = env.reset()

for i in range(23):
	ob,*_=env.step(1)
	env.render()

#import code
#code.interact(local=locals())

f = tf.zeros([7, 7]).eval()
f[3, 3] = 1
test(ob, f, 1, 'VALID', '7x7_identity_stride_1.png')
test(ob, f, 2, 'VALID', '7x7_identity_stride_2.png')
test(ob, f, 3, 'VALID', '7x7_identity_stride_3.png')
test(ob, f, 4, 'VALID', '7x7_identity_stride_4.png')
test(ob, f, 8, 'VALID', '7x7_identity_stride_8.png')

#import code
#code.interact(local=locals())
