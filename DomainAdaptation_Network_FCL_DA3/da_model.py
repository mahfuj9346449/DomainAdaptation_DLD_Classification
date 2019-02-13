import sys

sys.path.append('../Data_Initialization/')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import DomainAdaptation_Initialization as DA_init
import da_utils
import time


class DA_Model(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, restore_epoch, reloadPath, num_class,
                 learning_rate, lambda_para, batch_size, img_height, img_width, train_phase):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.source_validation_data = val_data[0]
        self.source_test_data = tst_data[0]
        self.target_training_data = train_data[1]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.res_eps = restore_epoch
        self.reloadPath = reloadPath
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.lr = learning_rate
        self.lbd = lambda_para
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase
        self.plt_epoch = []
        self.plt_training_accuracy = []
        self.plt_validation_accuracy = []
        self.plt_training_loss = []
        self.plt_validation_loss = []

        self.build_model()
        if self.train_phase == 'Train':
            self.saveConfiguration()

    def saveConfiguration(self):
        da_utils.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        da_utils.save2file('restore epoch : %d' % self.res_eps, self.ckptDir, self.model)
        da_utils.save2file('model : %s' % self.model, self.ckptDir, self.model)
        da_utils.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        da_utils.save2file('lambda : %g' % self.lbd, self.ckptDir, self.model)
        da_utils.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        da_utils.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        da_utils.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        da_utils.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        da_utils.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight',
                                          [ksize, ksize, inputMap.get_shape()[-1], out_channel],
                                          initializer=layers.variance_scaling_initializer())

            conv_result = tf.nn.conv2d(inputMap, conv_weight, strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def convTransposeLayer(self, inputMap, out_channel, ksize, stride, output_shape, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight',
                                          [ksize, ksize, out_channel, inputMap.get_shape().as_list()[-1]],
                                          initializer=layers.variance_scaling_initializer())

            conv_result = tf.nn.conv2d_transpose(inputMap, conv_weight, output_shape=output_shape,
                                                 strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def lreluLayer(self, inputMap, scope_name, alpha=0.2):
        with tf.variable_scope(scope_name):
            return tf.nn.leaky_relu(inputMap, alpha=alpha)

    def dropoutLayer(self, inputMap, scope_name, keep_rate):
        with tf.variable_scope(scope_name):
            return tf.nn.dropout(inputMap, keep_prob=keep_rate)

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

    def convBnReluLayer(self, inputMap, ksize, stride, out_channel, scope_name, is_training, use_bn, use_relu):
        if use_relu:
            activation = self.reluLayer
        else:
            activation = self.lreluLayer

        with tf.variable_scope(scope_name):
            _conv = self.convLayer(inputMap, out_channel=out_channel, ksize=ksize, stride=stride,
                                   scope_name='_conv')
            if use_bn:
                _conv = self.bnLayer(_conv, scope_name='_bn', is_training=is_training)
            _relu = activation(_conv, scope_name='_relu')

        return _relu

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

            _fm_bn = self.bnLayer(sec3_out, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax, [_relu, sec1_out, sec2_out, sec3_out]

    def featureConsistencyLayer(self, inputMapZoo, ksize, scope_name, is_training, adjustScale):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            input_shape = inputMapZoo[1].get_shape().as_list()
            output_shape = tf.shape(inputMapZoo[0])

            conv1 = self.convBnReluLayer(inputMapZoo[1],
                                         ksize=ksize,
                                         stride=1,
                                         out_channel=input_shape[-1],
                                         scope_name='conv',
                                         is_training=is_training,
                                         use_bn=True,
                                         use_relu=True)

            residual_section = self.residualSectionLayer(conv1,
                                                         ksize=ksize,
                                                         out_channel=input_shape[-1],
                                                         unit_num=3,
                                                         section_name='residual_section',
                                                         down_sampling=False,
                                                         first_conv=True,
                                                         is_training=is_training)

            if adjustScale:
                conv2 = self.convTransposeLayer(inputMap=residual_section,
                                                out_channel=input_shape[-1] // 2,
                                                ksize=3,
                                                stride=2,
                                                output_shape=output_shape,
                                                scope_name='conv_transpose')
            else:
                conv2 = self.convLayer(inputMap=residual_section,
                                       out_channel=input_shape[-1],
                                       ksize=3,
                                       stride=1,
                                       scope_name='conv_transpose')
            return conv2

    def Discriminator(self, inputMap, ksize, scope_name, is_training, keep_rate):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            in_channel = inputMap.get_shape()[-1]
            _layer1 = self.convBnReluLayer(inputMap, out_channel=in_channel, ksize=ksize, stride=1,
                                           scope_name='_layer1', is_training=is_training, use_bn=False, use_relu=False)
            _layer2 = self.convBnReluLayer(_layer1, out_channel=in_channel * 2, ksize=ksize, stride=1,
                                           scope_name='_layer2', is_training=is_training, use_bn=True, use_relu=False)
            _layer3 = self.convBnReluLayer(_layer2, out_channel=in_channel * 4, ksize=ksize, stride=1,
                                           scope_name='_layer3', is_training=is_training, use_bn=True, use_relu=False)
            _layer4 = self.convBnReluLayer(_layer3, out_channel=in_channel * 8, ksize=ksize, stride=1,
                                           scope_name='_layer4', is_training=is_training, use_bn=True, use_relu=False)

            _gap = self.globalPoolLayer(_layer4, scope_name='_gap')
            _flatten = self.flattenLayer(_gap, scope_name='_flatten')

            _dp1 = self.dropoutLayer(_flatten, scope_name='_dp1', keep_rate=keep_rate)
            _fc1 = self.fcLayer(_dp1, out_channel=512, scope_name='_fc1')
            _dp2 = self.dropoutLayer(_fc1, scope_name='_dp2', keep_rate=keep_rate)
            _fc2 = self.fcLayer(_dp2, out_channel=512, scope_name='_fc2')
            _dp3 = self.dropoutLayer(_fc2, scope_name='_dp3', keep_rate=keep_rate)
            _fc3 = self.fcLayer(_dp3, out_channel=1, scope_name='_fc3')

        return _fc3

    def build_model(self):
        self.x_source = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_source')
        self.x_target = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_target')

        self.y_source = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_source')
        self.y_target = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_target')

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_rate = tf.placeholder(tf.float32, name='keep_rate')

        tf.summary.image('source_input', self.x_source)
        tf.summary.image('target_input', self.x_target)

        self.g2_pred_source, self.g2_pred_softmax_source, self.g2_source_feature = self.resnet_model(
            input_x=self.x_source,
            model_name='G2',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64)

        self.g2_pred_target, self.g2_pred_softmax_target, self.g2_target_feature = self.resnet_model(
            input_x=self.x_target,
            model_name='G2',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64)

        # feature consistant in scale 3
        self.tar_fea_cons_sc3 = self.featureConsistencyLayer([self.g2_target_feature[2], self.g2_target_feature[3]],
                                                             ksize=3,
                                                             scope_name='FCL3',
                                                             is_training=self.is_training,
                                                             adjustScale=True)

        self.D_src_fea_sc3 = self.Discriminator(self.g2_source_feature[3],
                                                ksize=3,
                                                scope_name='D3',
                                                is_training=self.is_training,
                                                keep_rate=self.keep_rate)
        self.D_tar_fea_sc3 = self.Discriminator(self.g2_target_feature[3],
                                                ksize=3,
                                                scope_name='D3',
                                                is_training=self.is_training,
                                                keep_rate=self.keep_rate)

        with tf.variable_scope('loss'):
            # supervised loss
            self.supervised_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.g2_pred_source, labels=self.y_source))

            # feature consistency loss
            self.FCL_loss3 = tf.losses.absolute_difference(self.g2_target_feature[2],
                                                           self.tar_fea_cons_sc3)

            self.FCL_loss = self.FCL_loss3

            # generator losses
            self.g2_target_gloss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_tar_fea_sc3,
                labels=tf.ones_like(self.D_tar_fea_sc3)))

            self.g_target_loss = self.g2_target_gloss3

            self.g_loss = self.supervised_loss + self.lbd * self.g_target_loss

            # discriminator losses
            self.g2_source_dloss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_src_fea_sc3,
                labels=tf.ones_like(self.D_src_fea_sc3)))
            self.g2_target_dloss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_tar_fea_sc3,
                labels=tf.zeros_like(self.D_tar_fea_sc3)))

            self.d3_loss = self.g2_source_dloss3 + self.g2_target_dloss3

            self.d_loss = self.d3_loss

            tf.summary.scalar('supervised_loss', self.supervised_loss)
            tf.summary.scalar('g_loss3', self.g2_target_gloss3)
            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('d_loss3', self.d3_loss)
            tf.summary.scalar('d_loss', self.d_loss)
            tf.summary.scalar('FCL3_loss', self.FCL_loss3)
            tf.summary.scalar('FCL_LOSS', self.FCL_loss)

        with tf.variable_scope('optimization_variables'):
            self.t_var = tf.trainable_variables()

            self.g_var = [var for var in self.t_var if 'G2' in var.name]

            self.FCL3_var = [var for var in self.t_var if 'FCL3' in var.name]
            self.FCL_var = self.FCL3_var

            self.d3_var = [var for var in self.t_var if 'D3' in var.name]
            self.d_var = self.d3_var

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.FCL_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.FCL_loss,
                                                                             var_list=self.FCL_var)

                self.d_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss,
                                                                                      var_list=self.d_var)

                self.g_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss,
                                                                                      var_list=self.g_var)
        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            if self.train_phase == 'Train':
                self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution_source = [tf.argmax(self.y_source, 1), tf.argmax(self.g2_pred_softmax_source, 1)]
            self.distribution_target = [tf.argmax(self.y_target, 1), tf.argmax(self.g2_pred_softmax_target, 1)]

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

    def validation_procedure(self, validation_data, distribution_op, loss_op, inputX, inputY):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")
        val_loss = 0.0

        val_batch_num = int(np.ceil(validation_data[0].shape[0] / self.bs))
        for step in range(val_batch_num):
            _validationImg = validation_data[0][step * self.bs:step * self.bs + self.bs]
            _validationLab = validation_data[1][step * self.bs:step * self.bs + self.bs]

            [matrix_row, matrix_col], tmp_loss = self.sess.run([distribution_op, loss_op],
                                                               feed_dict={inputX: _validationImg,
                                                                          inputY: _validationLab,
                                                                          self.is_training: False,
                                                                          self.keep_rate: 0.5})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

            val_loss += tmp_loss

        validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        validation_loss = val_loss / val_batch_num

        return validation_accuracy, validation_loss

    def test_procedure(self, test_data, distribution_op, inputX, inputY, mode):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")

        tst_batch_num = int(np.ceil(test_data[0].shape[0] / self.bs))
        for step in range(tst_batch_num):
            _testImg = test_data[0][step * self.bs:step * self.bs + self.bs]
            _testLab = test_data[1][step * self.bs:step * self.bs + self.bs]

            matrix_row, matrix_col = self.sess.run(distribution_op, feed_dict={inputX: _testImg,
                                                                               inputY: _testLab,
                                                                               self.is_training: False,
                                                                               self.keep_rate: 0.5})
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
            log3 += 'category %s test accuracy : %g\n' % (da_utils.pulmonary_category[j], detail_test_accuracy[j])
        log3 = log3[:-1]
        log4 = 'F_Value : %g\n' % self.f_value(confusion_matrics)

        da_utils.save2file(log0, self.ckptDir, self.model)
        da_utils.save2file(log1, self.ckptDir, self.model)
        da_utils.save2file(log2, self.ckptDir, self.model)
        da_utils.save2file(log3, self.ckptDir, self.model)
        da_utils.save2file(log4, self.ckptDir, self.model)

    def getBatchData(self):
        _src_tr_img_batch, _src_tr_lab_batch = DA_init.next_batch(self.source_training_data[0],
                                                                  self.source_training_data[1], self.bs)
        _tar_tr_img_batch = DA_init.next_batch_unpaired(self.target_training_data, self.bs)

        feed_dict = {self.x_source: _src_tr_img_batch,
                     self.y_source: _src_tr_lab_batch,
                     self.x_target: _tar_tr_img_batch,
                     self.is_training: True,
                     self.keep_rate: 0.5}
        feed_dict_eval = {self.x_source: _src_tr_img_batch,
                          self.y_source: _src_tr_lab_batch,
                          self.x_target: _tar_tr_img_batch,
                          self.is_training: False,
                          self.keep_rate: 0.5}

        return feed_dict, feed_dict_eval

    def train(self):
        print('Start to run in mode [Domain Adaptation Across Source and Target Domain]')
        self.sess.run(tf.global_variables_initializer())
        self.reloadSaver = tf.train.Saver(var_list=self.g_var)
        self.reloadSaver.restore(self.sess, self.reloadPath)
        print('Pre-trained G2 model has been successfully reloaded !')

        self.itr_epoch = len(self.source_training_data[0]) // self.bs

        source_training_acc = 0.0
        source_training_loss = 0.0

        for e in range(1, self.eps + 1):
            for itr in range(self.itr_epoch):
                feed_dict_train, feed_dict_eval = self.getBatchData()
                _ = self.sess.run(self.FCL_train_op, feed_dict=feed_dict_train)
                _ = self.sess.run(self.d_train_op, feed_dict=feed_dict_train)

                feed_dict_train, feed_dict_eval = self.getBatchData()
                _ = self.sess.run(self.g_train_op, feed_dict=feed_dict_train)

                _training_accuracy, _training_loss = self.sess.run([self.accuracy_source, self.supervised_loss],
                                                                   feed_dict=feed_dict_eval)

                source_training_acc += _training_accuracy
                source_training_loss += _training_loss

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            source_training_acc = float(source_training_acc / self.itr_epoch)
            source_training_loss = float(source_training_loss / self.itr_epoch)

            source_validation_acc, source_validation_loss = self.validation_procedure(
                validation_data=self.source_validation_data, distribution_op=self.distribution_source,
                loss_op=self.supervised_loss, inputX=self.x_source, inputY=self.y_source)

            log1 = "Epoch: [%d], Domain: Source, Training Accuracy: [%g], Validation Accuracy: [%g], " \
                   "Training Loss: [%g], Validation Loss: [%g], Time: [%s]" % (
                       e, source_training_acc, source_validation_acc, source_training_loss, source_validation_loss,
                       time.ctime(time.time()))

            self.plt_epoch.append(e)
            self.plt_training_accuracy.append(source_training_acc)
            self.plt_training_loss.append(source_training_loss)
            self.plt_validation_accuracy.append(source_validation_acc)
            self.plt_validation_loss.append(source_validation_loss)

            da_utils.plotAccuracy(x=self.plt_epoch,
                                    y1=self.plt_training_accuracy,
                                    y2=self.plt_validation_accuracy,
                                    figName=self.model,
                                    line1Name='training',
                                    line2Name='validation',
                                    savePath=self.ckptDir)

            da_utils.plotLoss(x=self.plt_epoch,
                                y1=self.plt_training_loss,
                                y2=self.plt_validation_loss,
                                figName=self.model,
                                line1Name='training',
                                line2Name='validation',
                                savePath=self.ckptDir)

            da_utils.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

            self.test_procedure(self.source_test_data, distribution_op=self.distribution_source,
                                inputX=self.x_source,
                                inputY=self.y_source, mode='source')
            self.test_procedure(self.target_test_data, distribution_op=self.distribution_target,
                                inputX=self.x_target,
                                inputY=self.y_target, mode='target')

            source_training_acc = 0.0
            source_training_loss = 0.0

    def test(self):
        print('Start to run in mode [Test in Target Domain]')
        self.saver.restore(self.sess, self.ckptDir + self.model + '-' + str(self.res_eps))
        self.test_procedure(self.target_test_data, distribution_op=self.distribution_target, inputX=self.x_target,
                            inputY=self.y_target, mode='target')
