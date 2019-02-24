import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import tensorflow.contrib.layers as layers
import DomainAdaptation_Initialization as DA_init
import cyclegan_utils
import time


class CycleGAN_Model(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, num_class, learning_rate, batch_size,
                 img_height, img_width, train_phase):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.source_validation_data = val_data[0]
        self.source_test_data = tst_data[0]
        self.target_training_data = train_data[1]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.lr = learning_rate
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase

        self.build_cyclegan_model()
        if self.train_phase == 'Train':
            self.saveConfiguration()

    def saveConfiguration(self):
        cyclegan_utils.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        cyclegan_utils.save2file('model : %s' % self.model, self.ckptDir, self.model)
        cyclegan_utils.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        cyclegan_utils.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        cyclegan_utils.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        cyclegan_utils.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        cyclegan_utils.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        cyclegan_utils.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

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

    def maxpoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.max_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

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

    def Uet_G(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            conv1_1 = self.convBnReluLayer(inputMap, ksize=3, stride=1, out_channel=64, scope_name='conv1_1',
                                           is_training=is_training, use_bn=True, use_relu=False)
            conv1_2 = self.convBnReluLayer(conv1_1, ksize=3, stride=1, out_channel=64, scope_name='conv1_2',
                                           is_training=is_training, use_bn=True, use_relu=False)

            conv2_1 = self.convBnReluLayer(conv1_2, ksize=3, stride=2, out_channel=128, scope_name='conv2_1',
                                           is_training=is_training, use_bn=True, use_relu=False)
            conv2_2 = self.convBnReluLayer(conv2_1, ksize=3, stride=1, out_channel=128, scope_name='conv2_2',
                                           is_training=is_training, use_bn=True, use_relu=False)

            conv3_1 = self.convBnReluLayer(conv2_2, ksize=3, stride=2, out_channel=256, scope_name='conv3_1',
                                           is_training=is_training, use_bn=True, use_relu=False)
            conv3_2 = self.convBnReluLayer(conv3_1, ksize=3, stride=1, out_channel=256, scope_name='conv3_2',
                                           is_training=is_training, use_bn=True, use_relu=False)

            up1 = self.convTransposeLayer(conv3_2, out_channel=128, ksize=3, stride=2, output_shape=tf.shape(conv2_2),
                                          scope_name='up1')
            up1_bn = self.bnLayer(up1, scope_name='up1_bn', is_training=is_training)
            up1_cont = tf.concat([up1_bn, conv2_2], axis=3)
            up1_lrelu = self.lreluLayer(up1_cont, scope_name='up1_lrelu')

            up2 = self.convTransposeLayer(up1_lrelu, out_channel=64, ksize=3, stride=2, output_shape=tf.shape(conv1_2),
                                          scope_name='up2')
            up2_bn = self.bnLayer(up2, scope_name='up2_bn', is_training=is_training)
            up2_cont = tf.concat([up2_bn, conv1_2], axis=3)
            up2_lrelu = self.lreluLayer(up2_cont, scope_name='up2_lrelu')

            conv_final = self.convLayer(up2_lrelu, out_channel=1, ksize=3, stride=1, scope_name='conv_final')
            conv_final_act = tf.nn.tanh(conv_final, name='conv_final_act')

        return conv_final_act

    def Discriminator(self, inputMap, ksize, scope_name, is_training):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            _layer1 = self.convBnReluLayer(inputMap, out_channel=64, ksize=ksize, stride=1,
                                           scope_name='_layer1', is_training=is_training, use_bn=False, use_relu=False)
            _layer2 = self.convBnReluLayer(_layer1, out_channel=128, ksize=ksize, stride=2,
                                           scope_name='_layer2', is_training=is_training, use_bn=True, use_relu=False)
            _layer3 = self.convBnReluLayer(_layer2, out_channel=256, ksize=ksize, stride=2,
                                           scope_name='_layer3', is_training=is_training, use_bn=True, use_relu=False)
            _layer4 = self.convLayer(_layer3, out_channel=1, ksize=3, stride=1, scope_name='_layer4')

        return _layer4

    def build_cyclegan_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='Y')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.fake_y = self.Uet_G(inputMap=self.X, scope_name='G', is_training=self.is_training)
        self.fake_x = self.Uet_G(inputMap=self.Y, scope_name='F', is_training=self.is_training)

        self.rec_x = self.Uet_G(inputMap=self.fake_y, scope_name='F', is_training=self.is_training)
        self.rec_y = self.Uet_G(inputMap=self.fake_x, scope_name='G', is_training=self.is_training)

        self.real_y_dis = self.Discriminator(self.Y, ksize=3, scope_name='DY', is_training=self.is_training)
        self.fake_y_dis = self.Discriminator(self.fake_y, ksize=3, scope_name='DY', is_training=self.is_training)
        self.real_x_dis = self.Discriminator(self.X, ksize=3, scope_name='DX', is_training=self.is_training)
        self.fake_x_dis = self.Discriminator(self.fake_x, ksize=3, scope_name='DX', is_training=self.is_training)

        with tf.variable_scope('loss_functions'):
            # cycle_loss
            self.cycle_loss = 10 * tf.reduce_mean(tf.abs(self.rec_x - self.X)) + 10 * tf.reduce_mean(
                tf.abs(self.rec_y - self.Y))

            # X -> Y gan loss
            self.G_g_loss = tf.reduce_mean(tf.squared_difference(self.fake_y_dis, tf.ones_like(self.fake_y_dis)))
            self.G_loss = self.G_g_loss + self.cycle_loss
            self.DY_loss = (tf.reduce_mean(tf.squared_difference(self.real_y_dis, tf.ones_like(self.real_y_dis))) +
                            tf.reduce_mean(tf.squared_difference(self.fake_y_dis, tf.zeros_like(self.fake_y_dis)))) / 2

            # Y -> X gan loss
            self.F_g_loss = tf.reduce_mean(tf.squared_difference(self.fake_x_dis, tf.ones_like(self.fake_x_dis)))
            self.F_loss = self.F_g_loss + self.cycle_loss
            self.DX_loss = (tf.reduce_mean(tf.squared_difference(self.real_x_dis, tf.ones_like(self.real_x_dis))) +
                            tf.reduce_mean(tf.squared_difference(self.fake_x_dis, tf.zeros_like(self.fake_x_dis)))) / 2

            # Total Loss
            self.Total_G_loss = self.G_g_loss + self.F_g_loss + self.cycle_loss
            self.Total_D_loss = self.DX_loss + self.DY_loss

        tf.summary.scalar('loss/G', self.G_g_loss)
        tf.summary.scalar('loss/DY', self.DY_loss)
        tf.summary.scalar('loss/F', self.F_g_loss)
        tf.summary.scalar('loss/DX', self.DX_loss)
        tf.summary.scalar('loss/cycle', self.cycle_loss)
        tf.summary.scalar('loss/total_g', self.Total_G_loss)
        tf.summary.scalar('loss/total_d', self.Total_D_loss)

        tf.summary.image('X/origin', self.X, max_outputs=3)
        tf.summary.image('Y/origin', self.Y, max_outputs=3)
        tf.summary.image('X/generated', self.fake_x, max_outputs=3)
        tf.summary.image('X/reconstructed', self.rec_x, max_outputs=3)
        tf.summary.image('Y/generated', self.fake_y, max_outputs=3)
        tf.summary.image('Y/reconstructed', self.rec_y, max_outputs=3)

        with tf.variable_scope('optimization_variables'):
            self.t_var = tf.trainable_variables()

            self.G_var = [var for var in self.t_var if 'G' in var.name]
            self.F_var = [var for var in self.t_var if 'F' in var.name]

            self.DX_var = [var for var in self.t_var if 'DX' in var.name]
            self.DY_var = [var for var in self.t_var if 'DY' in var.name]

            self.generator_var = self.G_var + self.F_var
            self.discriminator_var = self.DX_var + self.DY_var

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.generator_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.Total_G_loss,
                                                                                             var_list=self.generator_var)
                self.discriminator_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.Total_D_loss,
                                                                                                 var_list=self.discriminator_var)

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

    def build_classification_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='Y')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.fake_y = self.Uet_G(inputMap=self.X, scope_name='G', is_training=self.is_training)
        self.fake_x = self.Uet_G(inputMap=self.Y, scope_name='F', is_training=self.is_training)

        self.rec_x = self.Uet_G(inputMap=self.fake_y, scope_name='F', is_training=self.is_training)
        self.rec_y = self.Uet_G(inputMap=self.fake_x, scope_name='G', is_training=self.is_training)

        self.real_y_dis = self.Discriminator(self.Y, ksize=3, scope_name='DY', is_training=self.is_training)
        self.fake_y_dis = self.Discriminator(self.fake_y, ksize=3, scope_name='DY', is_training=self.is_training)
        self.real_x_dis = self.Discriminator(self.X, ksize=3, scope_name='DX', is_training=self.is_training)
        self.fake_x_dis = self.Discriminator(self.fake_x, ksize=3, scope_name='DX', is_training=self.is_training)

        with tf.variable_scope('loss_functions'):
            # cycle_loss
            self.cycle_loss = 10 * tf.reduce_mean(tf.abs(self.rec_x - self.X)) + 10 * tf.reduce_mean(
                tf.abs(self.rec_y - self.Y))

            # X -> Y gan loss
            self.G_g_loss = tf.reduce_mean(tf.squared_difference(self.fake_y_dis, tf.ones_like(self.fake_y_dis)))
            self.G_loss = self.G_g_loss + self.cycle_loss
            self.DY_loss = (tf.reduce_mean(tf.squared_difference(self.real_y_dis, tf.ones_like(self.real_y_dis))) +
                            tf.reduce_mean(tf.squared_difference(self.fake_y_dis, tf.zeros_like(self.fake_y_dis)))) / 2

            # Y -> X gan loss
            self.F_g_loss = tf.reduce_mean(tf.squared_difference(self.fake_x_dis, tf.ones_like(self.fake_x_dis)))
            self.F_loss = self.F_g_loss + self.cycle_loss
            self.DX_loss = (tf.reduce_mean(tf.squared_difference(self.real_x_dis, tf.ones_like(self.real_x_dis))) +
                            tf.reduce_mean(tf.squared_difference(self.fake_x_dis, tf.zeros_like(self.fake_x_dis)))) / 2

            # Total Loss
            self.Total_G_loss = self.G_g_loss + self.F_g_loss + self.cycle_loss
            self.Total_D_loss = self.DX_loss + self.DY_loss

        tf.summary.scalar('loss/G', self.G_g_loss)
        tf.summary.scalar('loss/DY', self.DY_loss)
        tf.summary.scalar('loss/F', self.F_g_loss)
        tf.summary.scalar('loss/DX', self.DX_loss)
        tf.summary.scalar('loss/cycle', self.cycle_loss)
        tf.summary.scalar('loss/total_g', self.Total_G_loss)
        tf.summary.scalar('loss/total_d', self.Total_D_loss)

        tf.summary.image('X/origin', self.X, max_outputs=3)
        tf.summary.image('Y/origin', self.Y, max_outputs=3)
        tf.summary.image('X/generated', self.fake_x, max_outputs=3)
        tf.summary.image('X/reconstructed', self.rec_x, max_outputs=3)
        tf.summary.image('Y/generated', self.fake_y, max_outputs=3)
        tf.summary.image('Y/reconstructed', self.rec_y, max_outputs=3)

        with tf.variable_scope('optimization_variables'):
            self.t_var = tf.trainable_variables()

            self.G_var = [var for var in self.t_var if 'G' in var.name]
            self.F_var = [var for var in self.t_var if 'F' in var.name]

            self.DX_var = [var for var in self.t_var if 'DX' in var.name]
            self.DY_var = [var for var in self.t_var if 'DY' in var.name]

            self.generator_var = self.G_var + self.F_var
            self.discriminator_var = self.DX_var + self.DY_var

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.generator_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.Total_G_loss,
                                                                                             var_list=self.generator_var)
                self.discriminator_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.Total_D_loss,
                                                                                                 var_list=self.discriminator_var)

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

    def getBatchData(self):
        _src_tr_img_batch, _src_tr_lab_batch = DA_init.next_batch(self.source_training_data[0],
                                                                  self.source_training_data[1], self.bs, data_aug=False)
        _tar_tr_img_batch = DA_init.next_batch_unpaired(self.target_training_data, self.bs, data_aug=False)

        feed_dict = {self.X: _src_tr_img_batch,
                     self.Y: _tar_tr_img_batch,
                     self.is_training: True}

        feed_dict_eval = {self.X: _src_tr_img_batch,
                          self.Y: _tar_tr_img_batch,
                          self.is_training: False}

        return feed_dict, feed_dict_eval

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.itr_epoch = len(self.source_training_data[0]) // self.bs

        for e in range(1, self.eps + 1):
            for itr in range(self.itr_epoch):
                feed_dict_train, feed_dict_eval = self.getBatchData()
                _, _ = self.sess.run([self.generator_trainOp, self.discriminator_trainOp], feed_dict=feed_dict_train)

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            G_g_loss, F_g_loss, DX_loss, DY_loss, Cycle_loss = self.sess.run(
                [self.G_g_loss, self.F_g_loss, self.DX_loss, self.DY_loss, self.cycle_loss], feed_dict=feed_dict_eval)

            log1 = "Epoch: [%d], G_g_loss: [%g], F_g_loss: [%g], DX_loss: [%g], DY_loss: [%g], Cycle_loss: [%g], " \
                   "Time: [%s]" % (e, G_g_loss, F_g_loss, DX_loss, DY_loss, Cycle_loss, time.ctime(time.time()))

            cyclegan_utils.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))
