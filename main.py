from keras.datasets import mnist
from PIL import Image
import os


(train_image, train_label), (test_image, test_label) = mnist.load_data()

save_folder = "mnist_images"
for i in range(3_000):

    if i % 100 == 0:
        print(i)

    index = train_label[i]

    image = Image.fromarray(train_image[i])
    image_path = os.path.join(save_folder, f"{index}/{i}.png")
    image.save(image_path)