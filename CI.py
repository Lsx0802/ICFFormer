import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.linalg import orth
import cv2
import numpy as np
from skimage.transform import resize
import time
from iso import isolate
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = time.time()

tf.set_random_seed(4)  # the same output
name = "cnnsci-4"
name2 = "DMQ_Axial"
ps = "[new data" + name + " 117] "

m = 9
print(ps + str(m))
f = open('./' + str(m) + '/1' + str(name) + '_' + str(m) + ".txt", "w")
f2 = open('./' + str(m) + '/2' + str(name) + '_' + str(m) + ".txt", "w")
f3 = open('./' + str(m) + '/3' + str(name) + '_' + str(m) + ".txt", "w")
print(ps + str(m), file=f)

log_dir = './'
data_dir = '/home/public/Documents/Gummy/data/117newdata_2D3/' + str(m) + '/'

img_train, label_train = isolate.read_and_decode1(data_dir + name2 + "_trainData.tfrecords")
img_test, label_test = isolate.read_and_decode1(data_dir + name2 + "_testData.tfrecords")

label_train = tf.one_hot(indices=tf.cast(label_train, tf.int32), depth=2)
label_test = tf.one_hot(indices=tf.cast(label_test, tf.int32), depth=2)

img_batch_train, label_batch_train = tf.train.batch([img_train, label_train], batch_size=64, capacity=1000)
img_batch_test, label_batch_test = tf.train.batch([img_test, label_test], batch_size=40, capacity=40)


###################################################### Network ####################
sess = tf.InteractiveSession()
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 2])


x_image = tf.reshape(x, [-1, 28, 28, 1])
isolate.sp("x_image",x_image,1)

# -------------------------------------------------------------------------------------------------------------
# ---------------------------------- local  Layer  1------------------------------------
W_conv_1_local = weight_variable([5, 5, 1, 64])
b_conv_1_local = bias_variable([64])
h_conv_1_local = tf.nn.relu(conv2d(x_image, W_conv_1_local) + b_conv_1_local)
h_pool_1_local = max_pool_2x2(h_conv_1_local)

isolate.sp("h_conv_1_local",h_conv_1_local,64)
isolate.sp("h_pool_1_local",h_pool_1_local,64)

W_conv_2_local = weight_variable([5, 5, 64, 64])
b_conv_2_local = bias_variable([64])
h_conv_2_local = tf.nn.relu(conv2d(h_pool_1_local, W_conv_2_local) + b_conv_2_local)
h_pool_2_local = max_pool_2x2(h_conv_2_local)

isolate.sp("h_conv_2_local",h_conv_2_local,64)
isolate.sp("h_pool_2_local",h_pool_2_local,64)

h_pool22_1 = tf.reshape(h_pool_2_local, [-1, 7 * 7 * 64])

w_fc_11 = weight_variable([7 * 7 * 64, 500])
b_fc_11 = bias_variable([500])
fc_out_11 = tf.nn.relu(tf.matmul(h_pool22_1, w_fc_11) + b_fc_11)
# drop11 = tf.nn.dropout(fc_out_11, keep_prob)

w_fc_12 = weight_variable([500, 250])
b_fc_12 = bias_variable([250])
fc_out_12 = tf.nn.relu(tf.matmul(fc_out_11, w_fc_12) + b_fc_12)

# common_indevidual analysis for nonlocal feature extraction
common_V1 = weight_variable([250, 250])
bias_common1 = bias_variable([250])
common_feature1 = tf.nn.relu(tf.matmul(fc_out_12, common_V1) + bias_common1)

indevidal_Q1 = tf.placeholder(tf.float32, [250, 250])
bias_indevidual1 = bias_variable([250])
indevidual_feature1 = tf.nn.relu(tf.matmul(fc_out_12, indevidal_Q1) + bias_indevidual1)

# ---------------------------------- local  Layer  2------------------------------------
W_conv_3_local = weight_variable([5, 5, 1, 64])
b_conv_3_local = bias_variable([64])
h_conv_3_local = tf.nn.relu(conv2d(x_image, W_conv_3_local) + b_conv_3_local)
h_pool_3_local = max_pool_2x2(h_conv_3_local)

isolate.sp("h_conv_3_local",h_conv_3_local,64)
isolate.sp("h_pool_3_local",h_pool_3_local,64)

W_conv_4_local = weight_variable([5, 5, 64, 64])
b_conv_4_local = bias_variable([64])
h_conv_4_local = tf.nn.relu(conv2d(h_pool_3_local, W_conv_4_local) + b_conv_4_local)
h_pool_4_local = max_pool_2x2(h_conv_4_local)

isolate.sp("h_conv_4_local",h_conv_4_local,64)
isolate.sp("h_pool_4_local",h_pool_4_local,64)

h_pool_2_local = tf.reshape(h_pool_4_local, [-1, 7 * 7 * 64])

w_fc_21 = weight_variable([7 * 7 * 64, 500])
b_fc_21 = bias_variable([500])
fc_out_21 = tf.nn.relu(tf.matmul(h_pool_2_local, w_fc_21) + b_fc_21)
# drop11 = tf.nn.dropout(fc_out_11, keep_prob)

w_fc_22 = weight_variable([500, 250])
b_fc_22 = bias_variable([250])
fc_out_22 = tf.nn.relu(tf.matmul(fc_out_21, w_fc_22) + b_fc_22)

# common_indevidual analysis for nonlocal feature extraction
common_V2 = weight_variable([250, 250])
bias_common2 = bias_variable([250])
common_feature2 = tf.nn.relu(tf.matmul(fc_out_22, common_V2) + bias_common2)

indevidal_Q2 = tf.placeholder(tf.float32, [250, 250])
bias_indevidual2 = bias_variable([250])
indevidual_feature2 = tf.nn.relu(tf.matmul(fc_out_22, indevidal_Q2) + bias_indevidual2)
print(indevidual_feature2)


#----------------------------------------------    Fusion      ----------------------------------------------

feature_cat = tf.concat([common_feature1, common_feature2, indevidual_feature1, indevidual_feature2], 1)

print(feature_cat)
w_fc_f1 = weight_variable([1000, 500])
b_fc_f1 = bias_variable([500])
fc_f_out1 = tf.nn.relu(tf.matmul(feature_cat, w_fc_f1) + b_fc_f1)
drop1 = tf.nn.dropout(fc_f_out1, keep_prob)

w_fc_f2 = weight_variable([500, 2])
b_fc_f2 = bias_variable([2])
Sight = tf.matmul(drop1, w_fc_f2) + b_fc_f2
prediction = tf.nn.softmax(Sight)


with tf.name_scope('cross_entropy'):
    CL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Sight, labels=y_))
    tf.summary.scalar('CL', CL)

    fai = 0.5 / 4125
    thta = 0.5 / 250

    part1_1 = tf.norm((common_feature1 - common_feature2), 'fro', axis=(0, 1))
    # part1_2 = tf.norm((common_feature1 - common_feature3), 'fro', axis=(0, 1))
    # part1_3 = tf.norm((common_feature2 - common_feature3), 'fro', axis=(0, 1))

    part2_1 = tf.norm((fc_out_12 - tf.matmul(fc_out_12, tf.matmul(tf.transpose(common_V1), common_V1)) - tf.matmul(
        fc_out_12, tf.matmul(tf.transpose(indevidal_Q1), indevidal_Q1))), 'fro', axis=(0, 1))
    part2_2 = tf.norm((fc_out_22 - tf.matmul(fc_out_22, tf.matmul(tf.transpose(common_V2), common_V2)) - tf.matmul(
        fc_out_22, tf.matmul(tf.transpose(indevidal_Q2), indevidal_Q2))), 'fro', axis=(0, 1))
    # part2_3 = tf.norm((fc_out_32 - tf.matmul(fc_out_32, tf.matmul(tf.transpose(common_V3), common_V3)) - tf.matmul(
    #     fc_out_32, tf.matmul(tf.transpose(indevidal_Q3), indevidal_Q3))), 'fro', axis=(0, 1))

    part3_1 = tf.norm((tf.matmul(tf.transpose(common_V1), indevidal_Q1)), 'fro', axis=(0, 1))
    part3_2 = tf.norm((tf.matmul(tf.transpose(common_V2), indevidal_Q2)), 'fro', axis=(0, 1))
    # part3_3 = tf.norm((tf.matmul(tf.transpose(common_V3), indevidal_Q3)), 'fro', axis=(0, 1))
    # loss_CI = (part1_1 + part1_2 + part1_3) + fai * (part2_1 + part2_2 + part2_3) + thta * (part3_1 + part3_2 + part3_3)
    loss_CI = part1_1 + fai * (part2_1 + part2_2) + thta * (part3_1 + part3_2)

    cross_entropy = CL + loss_CI
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('LR'):
    learning_rate = tf.train.exponential_decay(5e-5, global_step, decay_steps=4125 / 64, decay_rate=0.98,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    tf.summary.scalar('learning rate', learning_rate)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        y_conv1_pre = tf.argmax(Sight, 1)
        y_conv2 = tf.argmax(y_, 1)
        predict = tf.equal(y_conv1_pre, y_conv2)
        correct_prediction = tf.equal(tf.argmax(Sight, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train/' + str(m) + '/' + str(name) + 'train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/tensorboard/' + str(m) + '/' + str(name) + '-' + str(m))
test_log = log_dir + '/tensorboard/' + str(m) + '/' + str(name) + '-' + str(m)

tf.global_variables_initializer().run()

# y_c = tf.reduce_sum(tf.multiply(Sight, y_), axis=1)
# target_conv_layer = alllll
# target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
# gb_grad = tf.gradients(cross_entropy, x_image)[0]

############################################ Training Layer ###############
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# threads = tf.train.start_queue_runners(sess=sess)

saver = tf.train.Saver()
for i in range(15001):
    global_step = i


    common_V1_r = sess.run(common_V1)
    common_V2_r = sess.run(common_V2)
    # common_V3_r = sess.run(common_V3)

    indevidal_Q1_r = orth(common_V1_r)
    indevidal_Q2_r = orth(common_V2_r)
    # indevidal_Q3_r = orth(common_V3_r)

    if indevidal_Q1_r.shape != (250, 250):
        n1 = indevidal_Q1_r.shape[1]
        Q1 = np.zeros((250, 250))
        Q1[:, 0:n1] = indevidal_Q1_r
        indevidal_Q1_r = Q1

    if indevidal_Q2_r.shape != (250, 250):
        n2 = indevidal_Q2_r.shape[1]
        Q2 = np.zeros((250, 250))
        Q2[:, 0:n2] = indevidal_Q2_r
        indevidal_Q2_r = Q2

    img_xs_DMQ_train, label_xs_DMQ_train = sess.run([img_batch_train, label_batch_train])
    img_xs_DMQ_test, label_xs_DMQ_test = sess.run([img_batch_test, label_batch_test])

    if i % 50 == 0:
        # summary, acc, learn, y_conv11, y_conv22, predict_r, image, target_convlayer, target_conv_layergrad, gbgrad = sess.run(
        #     [merged, accuracy, learning_rate, y_conv1_pre, y_conv2, predict, x_image, target_conv_layer,
        #      target_conv_layer_grad, gb_grad],
        summary, acc, learn, y_conv11, y_conv22, predict_r = sess.run(
            [merged, accuracy, learning_rate, y_conv1_pre, y_conv2, predict],
            feed_dict={x: img_xs_DMQ_test, y_: label_xs_DMQ_test,
                       indevidal_Q1: indevidal_Q1_r, indevidal_Q2: indevidal_Q2_r,
                       keep_prob: 1.0})
        sen, spe = isolate.sennspe(y_conv11, y_conv22, predict_r)
        # summary,acc,learn = sess.run([merged,accuracy,learning_rate],feed_dict={x:img_xs2,y_:label_xs2,keep_prob:1.0})
        test_writer.add_summary(summary, i)
        # print('Accuracy at step %d: %g  learn: %g' % (i,acc,learn))
        print('Accuracy at step %d: %g sen %f spe %f' % (i, acc, sen, spe), file=f)
        print('Accuracy at step %d: %g   learn:%g sen %f spe %f' % (i, acc, learn, sen, spe))
        test_writer.add_summary(summary, i)
        print('Data come from %d    Accuracy: %g    sen: %f  spe: %f' % (i, acc, sen, spe), file=f2)
        print('%g %f %f' % (acc, sen, spe), file=f3)
        print(y_conv22, file=f2)  # real input label
        print(y_conv11, file=f2)  # out_put    label
        print(predict_r, file=f2)  # predict    label
        # if (i == 0) or (i == 10000) or (i % 100 == 0):
        #     isolate.visualize(test_log, image, i, acc, target_convlayer, target_conv_layergrad, gbgrad)
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: img_xs_DMQ_train, y_: label_xs_DMQ_train,
                                             indevidal_Q1: indevidal_Q1_r, indevidal_Q2: indevidal_Q2_r,
                                             keep_prob: 0.5},
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            # saver.save(sess, log_dir + '/model.ckpt', i)
            print('Adding run metadata for ', i)
        else:
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: img_xs_DMQ_train, y_: label_xs_DMQ_train,
                                             indevidal_Q1: indevidal_Q1_r, indevidal_Q2: indevidal_Q2_r,
                                             keep_prob: 0.5})
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
end_time = time.time()

coord.request_stop()
coord.join(threads)
sess.close()

print('Total time is: ' + str((start_time - end_time) / 60) + ' minutes.')
print('Total time is: ' + str((start_time - end_time) / 60) + ' minutes.', file=f)

f.close()
f2.close()
f3.close()





