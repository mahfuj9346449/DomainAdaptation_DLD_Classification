import sys

sys.path.append('../Data_Initialization/')
import os
from FineTuning_model import ResNet
import DomainAdaptation_Initialization as DA_init
import argparse
import tensorflow as tf
import FineTuning_utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-train_phase', required=True, help='[whether to train or test the model]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')
parser.add_argument('-target_percentage', required=True, type=float, help='[set the percentage of target data for training]')
parser.add_argument('-data_domain', required=True, help='[set source data]')

parser.add_argument('-epoch', default=300, type=int)
parser.add_argument('-restore_epoch', default=0, type=int)
parser.add_argument('-num_class', default=6, type=int)
parser.add_argument('-ksize', default=3, type=int)
parser.add_argument('-out_channel1', default=16, type=int)
parser.add_argument('-out_channel2', default=32, type=int)
parser.add_argument('-out_channel3', default=64, type=int)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.data_domain == 'Source':
    src_name = 'source'
    tar_name = 'target'
elif args.data_domain == 'Target':
    src_name = 'target'
    tar_name = 'source'
else:
    src_name = ''
    tar_name = ''
    reloadPath = ''

src_training = DA_init.loadPickle(FineTuning_utils.experimentalPath, src_name + '_training.pkl')
src_validation = DA_init.loadPickle(FineTuning_utils.experimentalPath, src_name + '_validation.pkl')
src_test = DA_init.loadPickle(FineTuning_utils.experimentalPath, src_name + '_test.pkl')

tar_training = DA_init.loadPickle(FineTuning_utils.experimentalPath, tar_name + '_training.pkl')
tar_validation = DA_init.loadPickle(FineTuning_utils.experimentalPath, tar_name + '_validation.pkl')
tar_test = DA_init.loadPickle(FineTuning_utils.experimentalPath, tar_name + '_test.pkl')

src_training = FineTuning_utils.normalizeInput(src_training, mode='Paired')
src_validation = FineTuning_utils.normalizeInput(src_validation, mode='Paired')
src_test = FineTuning_utils.normalizeInput(src_test, mode='Paired')

tar_training = FineTuning_utils.normalizeInput(tar_training, mode='Paired')
tar_validation = FineTuning_utils.normalizeInput(tar_validation, mode='Paired')
tar_test = FineTuning_utils.normalizeInput(tar_test, mode='Paired')

# 取目标域数据集中指定百分比的数据参与训练
tar_training = FineTuning_utils.getPartialDataSet(tar_training, keep_percentage=args.target_percentage)

training = [np.concatenate((src_training[0], tar_training[0]), axis=0),
            np.concatenate((src_training[1], tar_training[1]), axis=0)]

validation = [np.concatenate((src_validation[0], tar_validation[0]), axis=0),
              np.concatenate((src_validation[1], tar_validation[1]), axis=0)]

print('source training image shape', str(src_training[0].shape))
print('source training label shape', src_training[1].shape)
print('source training image mean/std', str(src_training[0].mean()), str(src_training[0].std()))

print('source validation image shape', str(src_validation[0].shape))
print('source validation label shape', src_validation[1].shape)
print('source validation image mean/std', str(src_validation[0].mean()), str(src_validation[0].std()))

print('source test image shape', tar_test[0].shape)
print('source test label shape', tar_test[1].shape)
print('source test image mean/std', str(tar_test[0].mean()), str(tar_test[0].std()))

print('target training image shape', str(tar_training[0].shape))
print('target training label shape', tar_training[1].shape)
print('target training image mean/std', str(tar_training[0].mean()), str(tar_training[0].std()))

print('target validation image shape', str(tar_validation[0].shape))
print('target validation label shape', tar_validation[1].shape)
print('target validation image mean/std', str(tar_validation[0].mean()), str(tar_validation[0].std()))

print('target test image shape', tar_test[0].shape)
print('target test label shape', tar_test[1].shape)
print('target test image mean/std', str(tar_test[0].mean()), str(tar_test[0].std()))

print('training image shape', training[0].shape)
print('validation image shape', validation[0].shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    res_model = ResNet(model_name=args.model_name,
                       sess=sess,
                       train_data=training,
                       val_data=validation,
                       tst_data=[src_test, tar_test],
                       epoch=args.epoch,
                       restore_epoch=args.restore_epoch,
                       num_class=args.num_class,
                       ksize=args.ksize,
                       out_channel1=args.out_channel1,
                       out_channel2=args.out_channel2,
                       out_channel3=args.out_channel3,
                       learning_rate=args.learning_rate,
                       batch_size=args.batch_size,
                       img_height=args.img_height,
                       img_width=args.img_width,
                       train_phase=args.train_phase)

    if args.train_phase == 'Train':
        res_model.train()

    if args.train_phase == 'Test':
        res_model.test()
