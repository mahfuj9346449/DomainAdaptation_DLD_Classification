import sys

sys.path.append('../Data_Initialization/')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import DomainAdaptation_Initialization as DA_init
import ADDA_utils
import time


class DA_Model_step3(object):
    def __init__(self, model_name, sess, tst_data, epoch, restore_epoch, reloadPath, num_class, batch_size, img_height,
                 img_width, step):

        self.sess = sess
        self.source_test_data = tst_data[0]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.res_eps = restore_epoch
        self.reloadPath1 = reloadPath[0]
        self.reloadPath2 = reloadPath[1]
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.step = step
        self.plt_epoch = []
        self.plt_training_accuracy = []
        self.plt_validation_accuracy = []
        self.plt_training_loss = []
        self.plt_validation_loss = []

        self.build_model()
        self.saveConfiguration()

    def saveConfiguration(self):
        ADDA_utils.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        ADDA_utils.save2file('restore epoch : %d' % self.res_eps, self.ckptDir, self.model)
        ADDA_utils.save2file('model : %s' % self.model, self.ckptDir, self.model)
        ADDA_utils.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        ADDA_utils.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        ADDA_utils.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        ADDA_utils.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        ADDA_utils.save2file('step : %d' % self.step, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight',
                                          [ksize, ksize, inputMap.get_shape()[-1], out_channel],
                                          initializer=layers.variance_scaling_initializer())

            conv_result = tf.nn.conv2d(inputMap, conv_weight, strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def avgPoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalPoolLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgPoolLayer(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def flattenLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def fcLayer(self, inputMap, out_channel, scope_name):
        with tf.variable_scope(scope_name):
            in_channel = inputMap.get_shape()[-1]
            fc_weight = tf.get_variable('fc_weight', [in_channel, out_channel],
                                        initializer=layers.variance_scaling_initializer())
            fc_bias = tf.get_variable('fc_bias', [out_channel], initializer=tf.zeros_initializer())

            fc_result = tf.matmul(inputMap, fc_weight) + fc_bias

            tf.summary.histogram('fc_weight', fc_weight)
            tf.summary.histogram('fc_bias', fc_bias)
            tf.summary.histogram('fc_result', fc_result)

            return fc_result

    def residualUnitLayer(self, inputMap, out_channel, ksize, unit_name, down_sampling, is_training, first_conv=False):
        with tf.variable_scope(unit_name):
            in_channel = inputMap.get_shape().as_list()[-1]
            if down_sampling:
                stride = 2
                increase_dim = True
            else:
                stride = 1
                increase_dim = False

            if first_conv:
                conv_layer1 = self.convLayer(inputMap, out_channel, ksize, stride, scope_name='conv_layer1')
            else:
                bn_layer1 = self.bnLayer(inputMap, scope_name='bn_layer1', is_training=is_training)
                relu_layer1 = self.reluLayer(bn_layer1, scope_name='relu_layer1')
                conv_layer1 = self.convLayer(relu_layer1, out_channel, ksize, stride, scope_name='conv_layer1')

            bn_layer2 = self.bnLayer(conv_layer1, scope_name='bn_layer2', is_training=is_training)
            relu_layer2 = self.reluLayer(bn_layer2, scope_name='relu_layer2')
            conv_layer2 = self.convLayer(relu_layer2, out_channel, ksize, stride=1, scope_name='conv_layer2')

            if increase_dim:
                identical_mapping = self.avgPoolLayer(inputMap, ksize=2, stride=2, scope_name='identical_pool')
                identical_mapping = tf.pad(identical_mapping, [[0, 0], [0, 0], [0, 0],
                                                               [(out_channel - in_channel) // 2,
                                                                (out_channel - in_channel) // 2]])
            else:
                identical_mapping = inputMap

            added = tf.add(conv_layer2, identical_mapping)

            return added

    def residualSectionLayer(self, inputMap, ksize, out_channel, unit_num, section_name, down_sampling, first_conv,
                             is_training):
        with tf.variable_scope(section_name):
            _out = inputMap
            _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_1', down_sampling=down_sampling,
                                          first_conv=first_conv, is_training=is_training)
            for n in range(2, unit_num + 1):
                _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_' + str(n),
                                              down_sampling=False, first_conv=False, is_training=is_training)

            return _out

    def resnet_model(self, input_x, model_name, ksize, unit_num1, unit_num2, unit_num3, out_channel1, out_channel2,
                     out_channel3):
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            _conv = self.convLayer(input_x, out_channel1, ksize=ksize, stride=1, scope_name='unit1_conv')
            _bn = self.bnLayer(_conv, scope_name='unit1_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='unit1_relu')

            sec1_out = self.residualSectionLayer(inputMap=_relu,
                                                 ksize=ksize,
                                                 out_channel=out_channel1,
                                                 unit_num=unit_num1,
                                                 section_name='section1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            sec2_out = self.residualSectionLayer(inputMap=sec1_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel2,
                                                 unit_num=unit_num2,
                                                 section_name='section2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            sec3_out = self.residualSectionLayer(inputMap=sec2_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel3,
                                                 unit_num=unit_num3,
                                                 section_name='section3',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            return sec3_out

    def classifier(self, inputMap, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            _fm_bn = self.bnLayer(inputMap, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax

    def build_model(self):
        self.x_source = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_source')
        self.x_target = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_target')

        self.y_source = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_source')
        self.y_target = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_target')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        tf.summary.image('source_input', self.x_source)
        tf.summary.image('target_input', self.x_target)

        self.source_featureMaps = self.resnet_model(
            input_x=self.x_source,
            model_name='source_encoder',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64)

        self.target_featureMaps = self.resnet_model(
            input_x=self.x_target,
            model_name='target_encoder',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64)

        self.source_pred, self.source_pred_softmax = self.classifier(self.source_featureMaps, scope_name='classifier')
        self.target_pred, self.target_pred_softmax = self.classifier(self.target_featureMaps, scope_name='classifier')

        with tf.variable_scope('optimization_variables'):
            self.g_var = tf.global_variables()

            self.src_var = [var for var in self.g_var if 'source_encoder' in var.name]
            self.tar_var = [var for var in self.g_var if 'target_encoder' in var.name]
            self.cla_var = [var for var in self.g_var if 'classifier' in var.name]

        with tf.variable_scope('accuracy'):
            self.distribution_source = [tf.argmax(self.y_source, 1), tf.argmax(self.source_pred_softmax, 1)]
            self.distribution_target = [tf.argmax(self.y_target, 1), tf.argmax(self.target_pred_softmax, 1)]

            self.correct_prediction_source = tf.equal(self.distribution_source[0], self.distribution_source[1])
            self.correct_prediction_target = tf.equal(self.distribution_target[0], self.distribution_target[1])

            self.accuracy_source = tf.reduce_mean(tf.cast(self.correct_prediction_source, 'float'))
            self.accuracy_target = tf.reduce_mean(tf.cast(self.correct_prediction_target, 'float'))

    def f_value(self, matrix):
        f = 0.0
        length = len(matrix[0])
        for i in range(length):
            recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(self.num_class)])
            precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(self.num_class)])
            result = (recall * precision) / (recall + precision)
            f += result
        f *= (2 / self.num_class)

        return f

    def test_procedure(self, test_data, distribution_op, inputX, inputY, mode):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")

        tst_batch_num = int(np.ceil(test_data[0].shape[0] / self.bs))
        for step in range(tst_batch_num):
            _testImg = test_data[0][step * self.bs:step * self.bs + self.bs]
            _testLab = test_data[1][step * self.bs:step * self.bs + self.bs]

            matrix_row, matrix_col = self.sess.run(distribution_op, feed_dict={inputX: _testImg,
                                                                               inputY: _testLab,
                                                                               self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

        test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in
                                range(self.num_class)]
        log0 = "Mode: " + mode
        log1 = "Test Accuracy : %g" % test_accuracy
        log2 = np.array(confusion_matrics.tolist())
        log3 = ''
        for j in range(self.num_class):
            log3 += 'category %s test accuracy : %g\n' % (ADDA_utils.pulmonary_category[j], detail_test_accuracy[j])
        log3 = log3[:-1]
        log4 = 'F_Value : %g\n' % self.f_value(confusion_matrics)

        ADDA_utils.save2file(log0, self.ckptDir, self.model)
        ADDA_utils.save2file(log1, self.ckptDir, self.model)
        ADDA_utils.save2file(log2, self.ckptDir, self.model)
        ADDA_utils.save2file(log3, self.ckptDir, self.model)
        ADDA_utils.save2file(log4, self.ckptDir, self.model)

    def test(self):
        self.restore_src_encoder_saver = tf.train.Saver(var_list=self.src_var)
        self.restore_tar_encoder_saver = tf.train.Saver(var_list=self.tar_var)
        self.restore_classifier_saver = tf.train.Saver(var_list=self.cla_var)

        print('Reloading parameters')
        self.restore_src_encoder_saver.restore(self.sess, self.reloadPath1)
        print('source encoder reload finish')
        self.restore_classifier_saver.restore(self.sess, self.reloadPath1)
        print('classifier reload finish')
        self.restore_tar_encoder_saver.restore(self.sess, self.reloadPath2)
        print('target encoder reload finish')

        self.test_procedure(self.source_test_data, distribution_op=self.distribution_source, inputX=self.x_source,
                            inputY=self.y_source, mode='source')

        self.test_procedure(self.target_test_data, distribution_op=self.distribution_target, inputX=self.x_target,
                            inputY=self.y_target, mode='target')
