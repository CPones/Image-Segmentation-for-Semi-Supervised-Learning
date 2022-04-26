import os
import cv2
from tqdm import tqdm
import numpy as np
from paddle.vision import transforms as T
import paddle
from paddle.io import DataLoader

import argparse
from Unet import UNet
from PSPnet import PSPnet
from Deeplab import Deeplabv3
from dataset import ImageDataset

def parse_args():
	parser = argparse.ArgumentParser(description='Model Train')
	parser.add_argument('--test_images_path', type=str, default='data/val_image')
	parser.add_argument('--params_path', type=str, default='output')
	parser.add_argument('--checkpoint_path', type=str, default='output/final')
	parser.add_argument('--save_path', type=str, default='data/val_label')
	parser.add_argument('--model', type=str, default='UNet')#UNet  PSPnet  Deeplabv3
	parser.add_argument('--eval_num', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=4)
	return parser.parse_known_args()[0]


def get_path(image_path):
    files=[]
    for dir_name in os.listdir(image_path):
        for image_name in os.listdir(os.path.join(image_path,dir_name)):
            if image_name.endswith('.png') and not image_name.startswith('.'):
                files.append(os.path.join(image_path,dir_name,image_name))
    return sorted(files)

def get_test_data(test_images_path):
    test_data=[]
    for name in os.listdir(test_images_path):
        img_path=os.path.join(test_images_path,name)
        test_data.append(img_path)
    test_data=np.expand_dims(np.array(test_data),axis=1)
    return test_data

def main(args):
	test_data = get_test_data(args.test_images_path)

	eval_transform = T.Compose([
		T.Resize((224, 224)),
		T.Transpose(),
		T.Normalize(mean=0., std=255.)
	])

	test_dataset = ImageDataset(test_data, eval_transform)

	if args.model == 'UNet':
		model = paddle.Model(UNet())
	elif args.model == 'PSPnet':
		model = paddle.Model(PSPnet())
	elif args.model == 'Deeplabv3':
		model = paddle.Model(Deeplabv3())
	else:
		print('No model in this')

	model.load(args.checkpoint_path)
	for i, img in tqdm(enumerate(test_dataset)):
		img = paddle.to_tensor(img).unsqueeze(0)
		predict = np.array(model.predict_batch(img)).squeeze(0).squeeze(0)
		predict = predict.argmax(axis=0)
		image_path = test_dataset.path[i]
		path_lst = image_path[0].split("/")
		save_path = os.path.join(args.save_dir, path_lst[-1][:-5]) + ".jpg"
		cv2.imwrite(save_path, predict * 255)


if __name__ == '__main__':
	args = parse_args()
	print(args)
	main(args)
