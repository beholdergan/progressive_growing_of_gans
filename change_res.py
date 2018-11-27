import os
from PIL import Image

# select dataset folder to check and destination folder to put output images in
path = '../datasets/beauty_dataset/img/beauty_dataset'
dest_path = '../datasets/beauty_dataset/img/beauty_dataset_scaled'

# destination resolution
dest_res = 2 ** 8

for i, file in enumerate(os.listdir(path)):

    # open image using PIL to detect resolution.
    img = Image.open(os.path.join(path,file))
    width, height = img.size

    # resize image to destination resolution and save in dest folder
    img = img.resize((dest_res,dest_res),Image.ANTIALIAS)
    img.save(os.path.join(dest_path,file),quality=95)

    if i % 100 == 0:
        print("saved {}/{} images".format(i,len(os.listdir(path))))
