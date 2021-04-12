import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = []
i = 0
for img_path in glob.glob('dlib_data/*.jpg'):
    i += 1
    if i == 10:
        break
    images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)

plt.savefig('./te.jpg')