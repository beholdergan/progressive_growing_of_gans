import os
from PIL import Image

# select dataset to check.
path = '../datasets/beauty_dataset/img/beauty_dataset'

# initiate dictionary to store for each resolution, how many images are in folder.
resolutions = {}

for file in os.listdir(path):

    # open image using PIL to detect resolution.
    img = Image.open(os.path.join(path,file))
    width, height = img.size

    # prepare resolution key.
    size_key = "{}x{}".format(width,height)

    # increment counter if key's already in dict, otherwise add new key to the dict.
    if size_key in resolutions:
        resolutions[size_key] += 1
    else:
        resolutions[size_key] = 1

# print for each resolution, how many images there are.
for key,value in resolutions.items():
    print("number of images in size of {}: {}".format(key,value))
