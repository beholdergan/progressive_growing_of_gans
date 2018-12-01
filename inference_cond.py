import misc
import numpy as np

run_id = "022-pgan-beautydataset-cond-preset-v2-1gpu-fp32"
result_subdir = misc.create_result_subdir('results', 'inference_test')

network_pkl = misc.locate_network_pkl(run_id, snapshot)

if png_prefix is None:
    png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'

# initialize random seed
random_state = np.random.RandomState(random_seed)

#load network
network_pkl = misc.locate_network_pkl(run_id, snapshot)
print('Loading network from "%s"...' % network_pkl)
G, D, Gs = misc.load_network_pkl(run_id, None)

num_of_labels = 10

for i in range(num_of_labels):
    
    latents = misc.random_latents(1, Gs, random_state=random_state)
    labels = np.zeros([1, num_of_labels], np.float32)
    labels[0][i] = 1.0
    
    image = Gs.run(latents, labels, minibatch_size=1, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=1, out_dtype=np.uint8)
    
    #TODO: conver and save image
    
open(os.path.join(result_subdir, '_done.txt'), 'wt').close()