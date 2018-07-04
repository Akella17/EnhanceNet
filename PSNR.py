
import numpy as np
import math

def psnr(target, ref):
	#assume RGB image
	target_data = np.array(target)

	ref_data = np.array(ref)
	
	diff = ref_data - target_data
	diff = diff.flatten('C')
	rmse = math.sqrt( np.mean(diff ** 2.) )
	return 20*math.log10(1.0/rmse)
