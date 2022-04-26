import os
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
	parser.add_argument('--train_images_path', type=str, default='data/train_image')
	parser.add_argument('--label_images_path', type=str, default='data/train_50k_mask')
	parser.add_argument('--params_path', type=str, default='output')
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
	images = np.expand_dims(np.array(get_path(args.train_images_path)), axis=1)
	labels = np.expand_dims(np.array(get_path(args.label_images_path)), axis=1)
	data = np.array(np.concatenate((images, labels), axis=1))
	np.random.shuffle(data)
	train_data = data[:-args.eval_num, :]
	eval_data = data[-args.eval_num:, :]


	train_transform = T.Compose([
		T.Resize((224, 224)),  # 裁剪
		T.ColorJitter(0.1, 0.1, 0.1, 0.1),  # 亮度，对比度，饱和度和色调
		T.Transpose(),  # CHW
		T.Normalize(mean=0., std=255.)  # 归一化
	])
	eval_transform = T.Compose([
		T.Resize((224, 224)),
		T.Transpose(),
		T.Normalize(mean=0., std=255.)
	])

	train_dataset = ImageDataset(train_data, train_transform)
	eval_dataset = ImageDataset(eval_data, eval_transform)
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	if args.model == 'UNet':
		model = paddle.Model(UNet())
	elif args.model == 'PSPnet':
		model = paddle.Model(PSPnet())
	elif args.model == 'Deeplabv3':
		model = paddle.Model(Deeplabv3())
	else:
		print('No model in this')

	opt = paddle.optimizer.Momentum(learning_rate=1e-3, parameters=model.parameters(), weight_decay=1e-2)
	model.prepare(opt, paddle.nn.CrossEntropyLoss(axis=1))
	model.fit(train_dataloader, eval_dataloader, epochs=10, verbose=2, save_dir=args.params_path, log_freq=200)

if __name__ == '__main__':
	args = parse_args()
	print(args)
	main(args)
