import numpy as np
import gzip
import os
from Utils.dataset import one_hot_encoded
from Utils.download import download


base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

filename_x_train = "train-images-idx3-ubyte.gz"
filename_y_train = "train-labels-idx1-ubyte.gz"
filename_x_test = "t10k-images-idx3-ubyte.gz"
filename_y_test = "t10k-labels-idx1-ubyte.gz"

class MNIST:
    img_size = 28
    img_size_flat = img_size * img_size
    img_shape = (img_size, img_size)
    num_channels = 1
    img_shape_full = (img_size, img_size, num_channels)
    num_classes = 10

    def __init__(self, data_dir="data/MNIST/"):
        self.data_dir = data_dir

        self.num_train = 55000
        self.num_val = 5000
        self.num_test = 10000

        x_train = self._load_images(filename=filename_x_train)
        y_train_cls = self._load_cls(filename=filename_y_train)

        self.x_train = x_train[0:self.num_train] / 255.0
        self.x_val = x_train[self.num_train:] / 255.0
        self.y_train_cls = y_train_cls[0:self.num_train]
        self.y_val_cls = y_train_cls[self.num_train:]

        self.x_test = self._load_images(filename=filename_x_test) / 255.0
        self.y_test_cls = self._load_cls(filename=filename_y_test)

        self.y_train_cls = self.y_train_cls.astype(np.int)
        self.y_val_cls = self.y_val_cls.astype(np.int)
        self.y_test_cls = self.y_test_cls.astype(np.int)

        self.y_train = one_hot_encoded(class_numbers=self.y_train_cls,
                                       num_classes=self.num_classes)
        self.y_val = one_hot_encoded(class_numbers=self.y_val_cls,
                                     num_classes=self.num_classes)
        self.y_test = one_hot_encoded(class_numbers=self.y_test_cls,
                                      num_classes=self.num_classes)

    def _load_data(self, filename, offset):
        download(base_url=base_url, filename=filename, download_dir=self.data_dir)

        path = os.path.join(self.data_dir, filename)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)

        return data

    def _load_images(self, filename):
        data = self._load_data(filename=filename, offset=16)

        images_flat = data.reshape(-1, self.img_size_flat)

        return images_flat

    def _load_cls(self, filename):
        return self._load_data(filename=filename, offset=8)

    def random_batch(self, batch_size=32):
        idx = np.random.randint(low=0, high=self.num_train, size=batch_size)

        x_batch = self.x_train[idx]
        y_batch = self.y_train[idx]
        y_batch_cls = self.y_train_cls[idx]

        return x_batch, y_batch, y_batch_cls