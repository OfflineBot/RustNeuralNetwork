from keras.datasets import mnist
from PIL import Image
import os


(train_image, train_label), (test_image, test_label) = mnist.load_data()

save_folder = "mnist_images"
image_count = 30_000
for i in range(image_count):

    if i % 100 == 0:
        print(i)

    index = train_label[i]

    image = Image.fromarray(train_image[i])
    path = f"{save_folder}/{index}"
    number = os.listdir(path)
    image_path = os.path.join(save_folder, f"{index}/{len(number)}.png")
    image.save(image_path)
print(f"Added a total of {image_count} pictures")