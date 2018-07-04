from nnlib import *
import matplotlib.pyplot as plt

def method(recon_LR_img) :
	outputs = []
	PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
		xs = tf.placeholder(tf.float32, [1, 256, 256, 3])
		rblock = [resi, [[conv], [relu], [conv]]]
		ys_est = NN('generator',
		[xs,
		 [conv], [relu],
		 rblock, rblock, rblock, rblock, rblock,
		 rblock, rblock, rblock, rblock, rblock,
		 [upsample], [conv], [relu],
		 [upsample], [conv], [relu],
		 [conv], [relu],
		 [conv, 3]])
		ys_res = tf.image.resize_images(xs, [4*256, 4*256], method=tf.image.ResizeMethod.BICUBIC)
		ys_est += ys_res + PER_CHANNEL_MEANS
		sessi = tf.InteractiveSession()
		tf.train.Saver().restore(sessi, os.getcwd()+'/weights')
		for imgs in recon_LR_img:
		    imgs = np.expand_dims(imgs, axis=0)
		    imgsize = np.shape(imgs)[1:]		    
		    output = sessi.run([ys_est, ys_res+PER_CHANNEL_MEANS],
				      feed_dict={xs: imgs-PER_CHANNEL_MEANS})
		    plt.axis("off")
		    plt.imshow(output[0][0])
		    outputs.append(output)		    
		    #tf.reset_default_graph()
		sessi.close()
	return outputs
