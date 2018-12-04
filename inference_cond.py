import os
import misc
import numpy as np
import pdb
from config import EasyDict
import tfutil

# manual parameters
run_id = "022-pgan-beautydataset-cond-preset-v2-1gpu-fp32"
result_subdir = misc.create_result_subdir('results', 'inference_test')
num_of_examples = 500
num_of_labels = 10

misc.init_output_logging()

# initialize TensorFlow
print('Initializing TensorFlow...')
env = EasyDict() # Environment variables, set by the main program in train.py.
env.TF_CPP_MIN_LOG_LEVEL = '1' # Print warnings and errors, but disable debug info.
env.CUDA_VISIBLE_DEVICES = '1' # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use. change to '0' if first GPU is better
os.environ.update(env)
tf_config = EasyDict() # TensorFlow session config, set by tfutil.init_tf().
tf_config['graph_options.place_pruned_graph'] = True # False (default) = Check that all ops are available on the designated device.
tfutil.init_tf(tf_config)

#load network
network_pkl = misc.locate_network_pkl(run_id)
print('Loading network from "%s"...' % network_pkl)
G, D, Gs = misc.load_network_pkl(run_id, None)

# iterate random seed to generate [num_of_examples] examples of GIFs
for j in range(num_of_examples):

    random_seed = j # change it to control the generated latent vector

    # initialize random seed
    np.random.seed(random_seed)
    random_state = np.random.RandomState(random_seed)

    # generate random noise
    latents = misc.random_latents(1, Gs, random_state=random_state)

    for i in range(num_of_labels):
        
        # initiate conditioned label
        labels = np.zeros([1, num_of_labels], np.float32)
        labels[0][i] = 1.0
        
        # infer conditioned noise to receive image
        image = Gs.run(latents, labels, minibatch_size=1, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=1, out_dtype=np.uint8)

        # save generated image as 'i.png' and noise vector as noise_vector.txt
        misc.save_image_grid(image, os.path.join(result_subdir, '{}_{}.png'.format('%04d' % j,i)), [0,255], [1,1])

        # save latent space for later use
        np.save(os.path.join(result_subdir,'latents_vector.npy'), latents)

    if j % 10 == 0:
        print("saved {}/{} images".format(j,num_of_examples))

