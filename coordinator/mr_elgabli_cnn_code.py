import numpy as np
import tensorflow as tf
import os

list_of_npy_files = os.listdir()

index = 0
file_list_length = len(list_of_npy_files)
while index < file_list_length:
    if list_of_npy_files[index].endswith("npy"):
        if "combined" in list_of_npy_files[index]:
            list_of_npy_files.remove(list_of_npy_files[index])
            file_list_length = len(list_of_npy_files)
            continue
        else:
            index += 1
            continue
    else:
        list_of_npy_files.remove(list_of_npy_files[index])
        file_list_length = len(list_of_npy_files)
    pass

print(list_of_npy_files)

print(len(list_of_npy_files))

x_train = np.load('images_combined_worker.npy')
y_train = np.load('labels_combined_worker.npy')
x_test = np.load('images_combined_coordinator.npy')
y_test = np.load('labels_combined_coordinator.npy')

num_classes = 2
no_users = 6
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train / 255
x_test = x_test / 255

K = 4
L = 8
M = 12
N = 64

s1 = 4
s2 = 4

numClasses = 2


class User:
    def __init__(self):
        self.W1 = tf.Variable(tf.random.truncated_normal([s1, s2, 3, K], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([s1, s2, K, L], stddev=0.1))
        self.W3 = tf.Variable(tf.random.truncated_normal([s1, s2, L, M], stddev=0.1))
        self.W4 = tf.Variable(tf.random.truncated_normal([50 * 50 * M, N], stddev=0.1))
        self.W5 = tf.Variable(tf.random.truncated_normal([N, numClasses], stddev=0.1))

        self.b1 = tf.Variable(tf.random.truncated_normal([K]))  # tf.Variable(tf.ones([K])/10)
        self.b2 = tf.Variable(tf.random.truncated_normal([L]))  # tf.Variable(tf.ones([L])/10)
        self.b3 = tf.Variable(tf.random.truncated_normal([M]))  # tf.Variable(tf.ones([M])/10)
        self.b4 = tf.Variable(tf.random.truncated_normal([N]))  # tf.Variable(tf.ones([N])/10)
        self.b5 = tf.Variable(tf.random.truncated_normal([numClasses]))  # tf.Variable(tf.ones([numClasses])/10)

        self.gW1 = tf.Variable(tf.random.truncated_normal([s1, s2, 3, K], stddev=0.1))
        self.gW2 = tf.Variable(tf.random.truncated_normal([s1, s2, K, L], stddev=0.1))
        self.gW3 = tf.Variable(tf.random.truncated_normal([s1, s2, L, M], stddev=0.1))
        self.gW4 = tf.Variable(tf.random.truncated_normal([50 * 50 * M, N], stddev=0.1))
        self.gW5 = tf.Variable(tf.random.truncated_normal([N, numClasses], stddev=0.1))

        self.gb1 = tf.Variable(tf.random.truncated_normal([K]))  # tf.Variable(tf.ones([K])/10)
        self.gb2 = tf.Variable(tf.random.truncated_normal([L]))  # tf.Variable(tf.ones([L])/10)
        self.gb3 = tf.Variable(tf.random.truncated_normal([M]))  # tf.Variable(tf.ones([M])/10)
        self.gb4 = tf.Variable(tf.random.truncated_normal([N]))  # tf.Variable(tf.ones([N])/10)
        self.gb5 = tf.Variable(tf.random.truncated_normal([numClasses]))  # tf.Variable(tf.ones([numClasses])/10)

    def neural_net(self, x):
        C1 = tf.nn.conv2d(tf.reshape(x, [x.shape[0], 200, 200, 3]), self.W1, strides=[1, 1, 1, 1], padding="SAME")
        y1 = tf.nn.relu(C1 + self.b1)
        C2 = tf.nn.conv2d(y1, self.W2, strides=[1, 2, 2, 1], padding="SAME")
        y2 = tf.nn.relu(C2 + self.b2)
        C3 = tf.nn.conv2d(y2, self.W3, strides=[1, 2, 2, 1], padding="SAME")
        y3 = tf.nn.relu(C3 + self.b3)
        YY = tf.reshape(y3, shape=[-1, 50 * 50 * M])

        y4 = tf.nn.relu(tf.matmul(YY, self.W4) + self.b4)
        ylogits = tf.matmul(y4, self.W5)
        return tf.nn.softmax(ylogits + self.b5)


# mini-batch loss function.
def mini_batches(X, Y, mb_size):
    m = X.shape[0]
    perm = list(np.random.permutation(m))
    X_temp = X[perm, :]
    print(Y.shape)
    Y_temp = Y[perm, :].reshape((m, Y.shape[1]))

    X_r = tf.convert_to_tensor(X_temp[0:mb_size, :], dtype=np.float32)
    Y_r = tf.convert_to_tensor(Y_temp[0:mb_size, :], dtype=np.float32)
    print(X_r)
    print(Y_r)
    return X_r, Y_r


# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return -tf.reduce_sum(y_true * tf.math.log(y_pred))


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process.
def get_gradients(x, y, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5):
    # Variables to update, i.e. trainable variables.
    trainable_variables = [W1, W2, W3, W4, W5, b1, b2, b3, b4, b5]

    with tf.GradientTape() as g:
        C1 = tf.nn.conv2d(tf.reshape(x, [x.shape[0], 200, 200, 3]), W1, strides=[1, 1, 1, 1], padding="SAME")
        y1 = tf.nn.relu(C1 + b1)
        C2 = tf.nn.conv2d(y1, W2, strides=[1, 2, 2, 1], padding="SAME")
        y2 = tf.nn.relu(C2 + b2)
        C3 = tf.nn.conv2d(y2, W3, strides=[1, 2, 2, 1], padding="SAME")
        y3 = tf.nn.relu(C3 + b3)
        YY = tf.reshape(y3, shape=[-1, 50 * 50 * M])

        y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
        ylogits = tf.matmul(y4, W5)
        pred = tf.nn.softmax(ylogits + b5)
        loss = cross_entropy(pred, y)

        # Compute gradients.
    gradients1, gradients2, gradients3, gradients4, gradients5, gradients_b1, gradients_b2, gradients_b3, gradients_b4, gradients_b5 = g.gradient(
        loss, trainable_variables)

    return gradients1, gradients2, gradients3, gradients4, gradients5, gradients_b1, gradients_b2, gradients_b3, gradients_b4, gradients_b5, loss


users = [User() for i in range(no_users)]
#rho = 20
eta = 100
central_modal = [tf.Variable(tf.random.truncated_normal([s1*s2*3*K,1], stddev=0.1)), tf.Variable(tf.random.truncated_normal([s1*s2*K*L,1], stddev=0.1)), tf.Variable(tf.random.truncated_normal([s1*s2*L*M,1], stddev=0.1)), tf.Variable(tf.random.truncated_normal([50*50*M*N, 1], stddev=0.1)), tf.Variable(tf.random.truncated_normal([N*numClasses,1], stddev=0.1)), tf.Variable(tf.random.truncated_normal([K])), tf.Variable(tf.random.truncated_normal([L])), tf.Variable(tf.random.truncated_normal([M])), tf.Variable(tf.random.truncated_normal([N])), tf.Variable(tf.random.truncated_normal([numClasses]))]
x_train_k = []
y_train_k = []
data_per_worker = int(x_train.shape[0]/no_users)
print("Data per worker is: ", data_per_worker)
for i in range(no_users):
    first = i*data_per_worker
    last = first + data_per_worker
    x_train_k.append(x_train[first:last])
    y_train_k.append(y_train[first:last])

x_test = tf.convert_to_tensor(x_test, dtype=np.float32)
y_test = tf.convert_to_tensor(y_test, dtype=np.float32)

mb_size = 20
n_iters = 200
lr = 0.001
#n_localIter=1

Train_Acc = []
Test_Acc = []
acc_train = np.zeros([n_iters, 1])
acc_test = np.zeros([n_iters, 1])
total_loss = np.zeros([n_iters, 1])

for i in range(no_users):
    users[i].W1.assign(tf.reshape(central_modal[0], [s1, s2, 3, K]))
    users[i].W2.assign(tf.reshape(central_modal[1], [s1, s2, K, L]))
    users[i].W3.assign(tf.reshape(central_modal[2], [s1, s2, L, M]))
    users[i].W4.assign(tf.reshape(central_modal[3], [50 * 50 * M, N]))
    users[i].W5.assign(tf.reshape(central_modal[4], [N, numClasses]))

    users[i].b1.assign(tf.reshape(central_modal[5], [K]))
    users[i].b2.assign(tf.reshape(central_modal[6], [L]))
    users[i].b3.assign(tf.reshape(central_modal[7], [M]))
    users[i].b4.assign(tf.reshape(central_modal[8], [N]))
    users[i].b5.assign(tf.reshape(central_modal[9], [numClasses]))

for k in range(n_iters):
    batch_x = []
    batch_y = []
    for i in range(no_users):
        batch_xx, batch_yy = mini_batches(x_train_k[i], y_train_k[i], mb_size)
        batch_x.append(batch_xx)
        batch_y.append(batch_yy)
    for i in range(no_users):
        gradients1, gradients2, gradients3, gradients4, gradients5, gradients_b1, gradients_b2, gradients_b3, gradients_b4, gradients_b5, loss = get_gradients(
            batch_x[i], batch_y[i], users[i].W1, users[i].W2, users[i].W3, users[i].W4, users[i].W5, users[i].b1, users[i].b2, users[i].b3, users[i].b4,
            users[i].b5)

        users[i].gW1.assign(gradients1)
        users[i].gW2.assign(gradients2)
        users[i].gW3.assign(gradients3)
        users[i].gW4.assign(gradients4)
        users[i].gW5.assign(gradients5)

        users[i].gb1.assign(gradients_b1)
        users[i].gb2.assign(gradients_b2)
        users[i].gb3.assign(gradients_b3)
        users[i].gb4.assign(gradients_b4)
        users[i].gb5.assign(gradients_b5)

        total_loss[k] = total_loss[k] + loss

    temp11 = 0
    temp21 = 0
    temp31 = 0
    temp41 = 0
    temp51 = 0

    temp1_b = 0
    temp2_b = 0
    temp3_b = 0
    temp4_b = 0
    temp5_b = 0

    for i in range(no_users):
        temp11 = temp11 + tf.reshape(users[i].gW1, [s1 * s2 * 3 * K, 1])

        temp21 = temp21 + tf.reshape(users[i].gW2, [s1 * s2 * K * L, 1])

        temp31 = temp31 + tf.reshape(users[i].gW3, [s1 * s2 * L * M, 1])

        temp41 = temp41 + tf.reshape(users[i].gW4, [50 * 50 * M * N, 1])

        temp51 = temp51 + tf.reshape(users[i].gW5, [N * numClasses, 1])

        temp1_b = temp1_b + users[i].gb1  # tf.reshape(users[i].gb1,[K,1])

        temp2_b = temp2_b + users[i].gb2  # tf.reshape(users[i].gb2,[L,1])

        temp3_b = temp3_b + users[i].gb3  # tf.reshape(users[i].gb3,[M,1])

        temp4_b = temp4_b + users[i].gb4  # tf.reshape(users[i].gb4,[64,1])

        temp5_b = temp5_b + users[i].gb5  # tf.reshape(users[i].gb5,[10,1])

    # print(central_modal[5].shape[0])
    # print(central_modal[5].shape[1])
    # print("------")
    # Update central model
    central_modal[0] = central_modal[0] - 1 / (no_users) * lr * (temp11)
    central_modal[1] = central_modal[1] - 1 / (no_users) * lr * (temp21)
    central_modal[2] = central_modal[2] - 1 / (no_users) * lr * (temp31)
    central_modal[3] = central_modal[3] - 1 / (no_users) * lr * (temp41)
    central_modal[4] = central_modal[4] - 1 / (no_users) * lr * (temp51)

    central_modal[5] = central_modal[5] - 1 / (no_users) * lr * (temp1_b)
    central_modal[6] = central_modal[6] - 1 / (no_users) * lr * (temp2_b)
    central_modal[7] = central_modal[7] - 1 / (no_users) * lr * (temp3_b)
    central_modal[8] = central_modal[8] - 1 / (no_users) * lr * (temp4_b)
    central_modal[9] = central_modal[9] - 1 / (no_users) * lr * (temp5_b)

    # print(central_modal[5].shape[0])
    # print(central_modal[5].shape[1])

    for i in range(no_users):
        users[i].W1.assign(tf.reshape(central_modal[0], [s1, s2, 3, K]))
        users[i].W2.assign(tf.reshape(central_modal[1], [s1, s2, K, L]))
        users[i].W3.assign(tf.reshape(central_modal[2], [s1, s2, L, M]))
        users[i].W4.assign(tf.reshape(central_modal[3], [50 * 50 * M, N]))
        users[i].W5.assign(tf.reshape(central_modal[4], [N, numClasses]))

        users[i].b1.assign(tf.reshape(central_modal[5], [K]))
        users[i].b2.assign(tf.reshape(central_modal[6], [L]))
        users[i].b3.assign(tf.reshape(central_modal[7], [M]))
        users[i].b4.assign(tf.reshape(central_modal[8], [N]))
        users[i].b5.assign(tf.reshape(central_modal[9], [numClasses]))

    train_acc = []
    test_acc = []
    for j in range(no_users):
        train_pred = users[j].neural_net(batch_x[j])
        train_acc.append(accuracy(train_pred, batch_y[j]))
        test_pred = users[j].neural_net(x_test)
        test_acc.append(accuracy(test_pred, y_test))
    avgAcc_Train = np.mean(train_acc)
    avgAcc_Test = np.mean(test_acc)
    print('Train accuracy', avgAcc_Train)
    print('Test accuracy', avgAcc_Test)
    acc_train[k] = avgAcc_Train
    acc_test[k] = avgAcc_Test
