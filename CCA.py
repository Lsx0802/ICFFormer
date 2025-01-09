import tensorflow as tf
from scipy.linalg import orth
import time
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import os

i = '60'


def get_inputOp(filename, batch_size, capacity):
    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={"label": tf.FixedLenFeature([], tf.int64),
                                                     "image": tf.FixedLenFeature([], tf.string), })
        img = tf.decode_raw(features["image"], tf.int16)
        img = tf.reshape(img, [28 * 28 * 1])
        max = tf.to_float(tf.reduce_max(img))
        img = tf.cast(img, tf.float32) * (1.0 / max)
        label = tf.cast(features["label"], tf.int32)
        return img, label

    im, l = read_and_decode(filename)
    l = tf.one_hot(indices=tf.cast(l, tf.int32), depth=2)
    data, label = tf.train.batch([im, l], batch_size, capacity)
    return data, label


batch_size = 40

DMQ_Axial_dataTrain, DMQ_Axial_labelTrain = get_inputOp(
    "/home/public/june29/2D/DMQ/Axial/" + i + "/Train.tfrecords",
    batch_size, 1000)
DMQ_Axial_dataTest, DMQ_Axial_labelTest = get_inputOp(
    "/home/public/june29/2D/DMQ/Axial/" + i + "/Test.tfrecords",
    batch_size, batch_size)
DMQ_Coronal_dataTrain, DMQ_Coronal_labelTrain = get_inputOp(
    "/home/public/june29/2D/MMQ/Axial/" + i + "/Train.tfrecords",
    batch_size, 1000)
DMQ_Coronal_dataTest, DMQ_Coronal_labelTest = get_inputOp(
    "/home/public/june29/2D/MMQ/Axial/" + i + "/Test.tfrecords",
    batch_size, batch_size)
DMQ_Sagittal_dataTrain, DMQ_Sagittal_labelTrain = get_inputOp(
    "/home/public/june29/2D/PS/Axial/" + i + "/Train.tfrecords",
    batch_size, 1000)
DMQ_Sagittal_dataTest, DMQ_Sagittal_labelTest = get_inputOp(
    "/home/public/june29/2D/PS/Axial/" + i + "/Test.tfrecords",
    batch_size, batch_size)

sess = tf.InteractiveSession()
global_step = tf.Variable(0)
keep_prob = tf.placeholder(tf.float32)


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


def precision(model_output, equal):
    positive_position = 1
    negative_position = 0
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(equal)):
        if equal[i] == True:
            staticity_T[model_output[i]] += 1
        else:
            staticity_F[model_output[i]] += 1
    precision = staticity_T[positive_position] / (
            staticity_T[positive_position] + staticity_F[(negative_position + 1) % 2])
    return precision


def Sensitivity_specificity(model_output, equal):
    positive_position = 1
    negative_position = 0
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(equal)):
        if equal[i] == True:
            staticity_T[model_output[i]] += 1
        else:
            staticity_F[model_output[i]] += 1

    sensitivity = staticity_T[positive_position] / (
            staticity_T[positive_position] + staticity_F[(positive_position + 1) % 2])
    specificity = staticity_T[negative_position] / (
            staticity_T[negative_position] + staticity_F[(negative_position + 1) % 2])
    return sensitivity, specificity


a1 = tf.Variable(0.34)
a2 = tf.Variable(0.33)
a3 = 1 - a1 - a2

######################################   XOY_Frature   ###################################

x1 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
label1 = tf.placeholder(tf.float32, [None, 2])

inputData_1 = tf.reshape(x1, [-1, 28, 28, 1])

kernel_11 = weight_variable([3, 3, 1, 32])
bias_11 = bias_variable([32])
conv_11 = conv2d(inputData_1, kernel_11)
conv_out_11 = tf.nn.relu(conv_11 + bias_11)
pooling_out_11 = max_pool_2x2(conv_out_11)

kernel_12 = weight_variable([3, 3, 32, 64])
bias_12 = bias_variable([64])
conv_12 = conv2d(pooling_out_11, kernel_12)
conv_out_12 = tf.nn.relu(conv_12 + bias_12)
pooling_out_12 = max_pool_2x2(conv_out_12)

pooling_out_12 = tf.reshape(pooling_out_12, [-1, 7 * 7 * 64])

w_fc_11 = weight_variable([7 * 7 * 64, 500])
b_fc_11 = bias_variable([500])
fc_out_11 = tf.nn.relu(tf.matmul(pooling_out_12, w_fc_11) + b_fc_11)
drop11 = tf.nn.dropout(fc_out_11, keep_prob)

w_fc_12 = weight_variable([500, 50])
b_fc_12 = bias_variable([50])
fc_out_12 = tf.nn.relu(tf.matmul(drop11, w_fc_12) + b_fc_12)

w_fc_13 = weight_variable([500, 50])
b_fc_13 = bias_variable([50])
fc_out_13 = tf.nn.relu(tf.matmul(drop11, w_fc_13) + b_fc_13)
######################################   XOZ_Feature   ###################################

x2 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])

inputData_2 = tf.reshape(x2, [-1, 28, 28, 1])

kernel_21 = weight_variable([3, 3, 1, 32])
bias_21 = bias_variable([32])
conv_21 = conv2d(inputData_2, kernel_21)
conv_out_21 = tf.nn.relu(conv_21 + bias_21)
pooling_out_21 = max_pool_2x2(conv_out_21)

kernel_22 = weight_variable([3, 3, 32, 64])
bias_22 = bias_variable([64])
conv_22 = conv2d(pooling_out_21, kernel_22)
conv_out_22 = tf.nn.relu(conv_22 + bias_22)
pooling_out_22 = max_pool_2x2(conv_out_22)

pooling_out_22 = tf.reshape(pooling_out_22, [-1, 7 * 7 * 64])

w_fc_21 = weight_variable([7 * 7 * 64, 500])
b_fc_21 = bias_variable([500])
fc_out_21 = tf.nn.relu(tf.matmul(pooling_out_22, w_fc_21) + b_fc_21)
drop21 = tf.nn.dropout(fc_out_21, keep_prob)

w_fc_22 = weight_variable([500, 50])
b_fc_22 = bias_variable([50])
fc_out_22 = tf.nn.relu(tf.matmul(drop21, w_fc_22) + b_fc_22)

w_fc_23 = weight_variable([500, 50])
b_fc_23 = bias_variable([50])
fc_out_23 = tf.nn.relu(tf.matmul(drop21, w_fc_23) + b_fc_23)

######################################   YOZ_Feature   ###################################

x3 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])

inputData_3 = tf.reshape(x3, [-1, 28, 28, 1])

kernel_31 = weight_variable([3, 3, 1, 32])
bias_31 = bias_variable([32])
conv_31 = conv2d(inputData_3, kernel_31)
conv_out_31 = tf.nn.relu(conv_31 + bias_31)
pooling_out_31 = max_pool_2x2(conv_out_31)

kernel_32 = weight_variable([3, 3, 32, 64])
bias_32 = bias_variable([64])
conv_32 = conv2d(pooling_out_31, kernel_32)
conv_out_32 = tf.nn.relu(conv_32 + bias_32)
pooling_out_32 = max_pool_2x2(conv_out_32)

pooling_out_32 = tf.reshape(pooling_out_32, [-1, 7 * 7 * 64])

w_fc_31 = weight_variable([7 * 7 * 64, 500])
b_fc_31 = bias_variable([500])
fc_out_31 = tf.nn.relu(tf.matmul(pooling_out_32, w_fc_31) + b_fc_31)
drop31 = tf.nn.dropout(fc_out_31, keep_prob)

w_fc_32 = weight_variable([500, 50])
b_fc_32 = bias_variable([50])
fc_out_32 = tf.nn.relu(tf.matmul(drop31, w_fc_32) + b_fc_32)

w_fc_33 = weight_variable([500, 50])
b_fc_33 = bias_variable([50])
fc_out_33 = tf.nn.relu(tf.matmul(drop31, w_fc_33) + b_fc_33)
################################################################
concat1 = tf.concat([fc_out_12, fc_out_22, fc_out_32], 1)

w_fusion_11 = weight_variable([150, 50])
b_fusion_11 = bias_variable([50])
w_fusion_21 = weight_variable([150, 50])
b_fusion_21 = bias_variable([50])
w_fusion_31 = weight_variable([150, 50])
b_fusion_31 = bias_variable([50])

alpha11 = tf.sigmoid(tf.matmul(concat1, w_fusion_11) + b_fusion_11)
alpha21 = tf.sigmoid(tf.matmul(concat1, w_fusion_21) + b_fusion_21)
alpha31 = tf.sigmoid(tf.matmul(concat1, w_fusion_31) + b_fusion_31)

f11 = tf.multiply(alpha11, fc_out_12)
f21 = tf.multiply(alpha21, fc_out_22)
f31 = tf.multiply(alpha31, fc_out_32)

fc11 = tf.concat([f11, f21, f31],1)

###########################################
concat2 = tf.concat([fc_out_13, fc_out_23, fc_out_33], 1)

w_fusion_12 = weight_variable([150, 50])
b_fusion_12 = bias_variable([50])
w_fusion_22 = weight_variable([150, 50])
b_fusion_22 = bias_variable([50])
w_fusion_32 = weight_variable([150, 50])
b_fusion_32 = bias_variable([50])

alpha12 = tf.sigmoid(tf.matmul(concat2, w_fusion_12) + b_fusion_12)
alpha22 = tf.sigmoid(tf.matmul(concat2, w_fusion_22) + b_fusion_22)
alpha32 = tf.sigmoid(tf.matmul(concat2, w_fusion_32) + b_fusion_32)

f12 = tf.multiply(alpha12, fc_out_13)
f22 = tf.multiply(alpha22, fc_out_23)
f32 = tf.multiply(alpha32, fc_out_33)

fc21 = tf.concat([f12, f22, f32],1)

#########################################
concat3 = tf.concat([fc11,fc21], 1)

w_fusion_13 = weight_variable([300, 150])
b_fusion_13 = bias_variable([150])
w_fusion_23 = weight_variable([300, 150])
b_fusion_23 = bias_variable([150])

alpha13 = tf.sigmoid(tf.matmul(concat3, w_fusion_13) + b_fusion_13)
alpha23 = tf.sigmoid(tf.matmul(concat3, w_fusion_23) + b_fusion_23)

fc12 = tf.multiply(alpha13, fc11)
fc22 = tf.multiply(alpha23, fc21)

##########################################
rm1 = tf.reduce_mean(fc11)
rm2 = tf.reduce_mean(fc21)

f4 = tf.stack([rm1, rm2])
beta = tf.nn.softmax(f4)

b1 = beta[0]
b2= beta[1]

###################################### common DSN
w_fc_13 = weight_variable([150, 2])
b_fc_13 = bias_variable([2])
mid1 = tf.matmul(fc11, w_fc_13) + b_fc_13

with tf.name_scope('Loss_common'):
    L1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid1, labels=label1))
    tf.summary.scalar('Loss_common', L1)

################### individual Classify #############################
w_fc_23 = weight_variable([150, 2])
b_fc_23 = bias_variable([2])
mid2 = tf.matmul(fc21, w_fc_23) + b_fc_23

with tf.name_scope('Loss_individual'):
    L2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid2, labels=label1))
    tf.summary.scalar('Loss_individual', L2)
######################################     Fusion      ####################################
feature_cat = tf.concat([fc12, fc22], 1)

w_fc_f1 = weight_variable([300, 50])
b_fc_f1 = bias_variable([50])
fc_f_out1 = tf.nn.relu(tf.matmul(feature_cat, w_fc_f1) + b_fc_f1)
drop1 = tf.nn.dropout(fc_f_out1, keep_prob)

w_fc_f2 = weight_variable([50, 2])
b_fc_f2 = bias_variable([2])
mid = tf.matmul(fc_f_out1, w_fc_f2) + b_fc_f2
prediction = tf.nn.softmax(mid)

with tf.name_scope('loss'):
    CL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid, labels=label1))
    tf.summary.scalar('CL', CL)

    X = fc_out_11 - tf.reduce_mean(fc_out_11, axis=0, keepdims=True)
    Y = fc_out_21 - tf.reduce_mean(fc_out_21, axis=0, keepdims=True)
    Z = fc_out_31 - tf.reduce_mean(fc_out_31, axis=0, keepdims=True)

    up1 = tf.reduce_sum(X * Y)
    down1 = tf.sqrt(tf.reduce_sum(tf.reduce_sum(X * X, 0) * tf.reduce_sum(Y * Y, 0)))

    up2 = tf.reduce_sum(X * Z)
    down2 = tf.sqrt(tf.reduce_sum(tf.reduce_sum(X * X, 0) * tf.reduce_sum(Z * Z, 0)))

    up3 = tf.reduce_sum(Y * Z)
    down3 = tf.sqrt(tf.reduce_sum(tf.reduce_sum(Y * Y, 0) * tf.reduce_sum(Z * Z, 0)))

    L_1 = up1 / down1
    L_2 = up2 / down2
    L_3 = up3 / down3

    total_loss = CL + a1 * L_1 + a2 * L_2 + a3 * L_3 + b1 * L1 + b2 * L2

    tf.summary.scalar('total_loss', total_loss)

with tf.name_scope('LR'):
    learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=64, decay_rate=0.92,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    tf.summary.scalar('learning rate', learning_rate)

with tf.name_scope('Accuracy'):
    output_position = tf.argmax(prediction, 1)
    label_position = tf.argmax(label1, 1)
    predict = tf.equal(output_position, label_position)
    Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
    tf.summary.scalar('Accuracy', Accuracy)

merge = tf.summary.merge_all()
#####################################    Train    ##########################################

sess.run(tf.global_variables_initializer())

board_path = '/home/public/june29/log/' + '3D'
if (not (os.path.exists(board_path))):
    os.mkdir(board_path)

board_path = board_path + '/' + '1e-4_64_0.92'
if (not (os.path.exists(board_path))):
    os.mkdir(board_path)

board_path = board_path + '/' + i
if (not (os.path.exists(board_path))):
    os.mkdir(board_path)

image_board_path = board_path + '/' + 'image'
if (not (os.path.exists(image_board_path))):
    os.mkdir(image_board_path)

test_board_path = board_path + '/' + 'test'
if (not (os.path.exists(test_board_path))):
    os.mkdir(test_board_path)

train_board_path = board_path + '/' + 'train'
if (not (os.path.exists(train_board_path))):
    os.mkdir(train_board_path)

test_writer = tf.summary.FileWriter(test_board_path + '/', tf.get_default_graph())
train_writer = tf.summary.FileWriter(train_board_path + '/', tf.get_default_graph())

tf.train.start_queue_runners(sess=sess)

before = time.time()
accmax = 0

b1_=[]
b2_=[]
L1_=[]
L2_=[]
acc_=[]
loss_=[]

for times in range(10000):
    global_step = times

    DMQ_Axial_dataTest_r, DMQ_Coronal_dataTest_r, DMQ_Sagittal_dataTest_r, \
    DMQ_Axial_labelTest_r, DMQ_Coronal_labelTest_r, DMQ_Sagittal_labelTest_r = sess.run(
        [DMQ_Axial_dataTest, DMQ_Coronal_dataTest, DMQ_Sagittal_dataTest, DMQ_Axial_labelTest,
         DMQ_Coronal_labelTest, DMQ_Sagittal_labelTest])
    DMQ_Axial_dataTrain_r, DMQ_Coronal_dataTrain_r, DMQ_Sagittal_dataTrain_r, \
    DMQ_Axial_labelTrain_r, DMQ_Coronal_labelTrain_r, DMQ_Sagittal_labelTrain_r = sess.run(
        [DMQ_Axial_dataTrain, DMQ_Coronal_dataTrain, DMQ_Sagittal_dataTrain,
         DMQ_Axial_labelTrain, DMQ_Coronal_labelTrain, DMQ_Sagittal_labelTrain])

    ###########################  test  #######################
    if times % 10 == 0:
        summary, acc, output_position_r, label_position_r, predict_r, p, a1r, a2r, a3r,b1r,b2r,L1r,L2r,CLr = sess.run(
            [merge, Accuracy, output_position, label_position, predict, prediction, a1, a2, a3,b1,b2,L1,L2,CL],
            feed_dict={x1: DMQ_Axial_dataTest_r, x2: DMQ_Coronal_dataTest_r,
                       x3: DMQ_Sagittal_dataTest_r, label1: DMQ_Axial_labelTest_r, keep_prob: 1.0})

        sen, spe = Sensitivity_specificity(output_position_r, predict_r)
        fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        b1_.append(b1r)
        b2_.append(b2r)
        L1_.append(L1r)
        L2_.append(L2r)
        acc_.append(acc)
        loss_.append(CL)

        if acc >= accmax:
            accmax = acc
            print(times, ':', accmax)
            print(b1r,b2r)

    ###########################  show  #######################
    if times == 9999:
        pre = precision(output_position_r, predict_r)
        print('precision is ' + str(pre))
        fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
        AUC = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic_2D_CCA')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % AUC)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


    ###########################  train  #######################
    if times % 99 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merge, train_step],
                              feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
                                         x3: DMQ_Sagittal_dataTrain_r,
                                         label1: DMQ_Axial_labelTrain_r, keep_prob: 0.5})
        train_writer.add_run_metadata(run_metadata, 'step%03d' % times)
        train_writer.add_summary(summary, times)
    else:
        sess.run([train_step], feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
                                          x3: DMQ_Sagittal_dataTrain_r,
                                          label1: DMQ_Axial_labelTrain_r, keep_prob: 0.5})

np.savez(image_board_path,b1_=b1_,b2_=b2_,L1_=L1_,L2_=L2_,acc=acc_,loss_=loss_)
after = time.time()
print('Total time is: ' + str((after - before) / 60) + ' minutes.')
train_writer.close()
test_writer.close()
