import cv2
import numpy as np
from paddle.io import Dataset

class ImageDataset(Dataset):
	def __init__(self, path, transform):
		super(ImageDataset, self).__init__()
		self.path = path
		self.transform = transform

	def _load_image(self, path):
		img = cv2.imread(path)
		img = cv2.resize(img, (224, 224))
		return img

	def __getitem__(self, index):
		path = self.path[index]
		if len(path) == 2:
			data_path, label_path = path
			data = self._load_image(data_path)
			label = self._load_image(label_path)
			data, label = self.transform(data), label
			label = label.transpose((2, 0, 1))
			label = label[0, :, :]
			label = np.expand_dims(label, axis=0)
			if True in (label>1):
				label = label/255.
			label = label.astype("int64")
			return data, label

		if len(path) == 1:
			data = self._load_image(path[0])
			data = self.transform(data)
			return data

	def __len__(self):
		return len(self.path)