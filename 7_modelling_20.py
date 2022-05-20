from numpy import loadtxt
from numpy import savetxt
import numpy
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow

def reshape_function(row):
    return row.reshape(5, 4).transpose().reshape(1, 20)

# setting the seed
seed(1)
set_seed(1)

index1 = 4241

rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

orig_epochs = loadtxt('my_epochs.csv', delimiter=',', skiprows=1)
print(orig_epochs.shape)

my_epochs = (numpy.apply_along_axis(reshape_function, 1, orig_epochs[:, 0:20])).reshape(index1, 20)
my_epochs = numpy.append(my_epochs, orig_epochs[:, -1].reshape(len(orig_epochs[:, -1]), 1), axis=1)

print(my_epochs.shape)

my_epochs_1 = rScaler.fit_transform(my_epochs[0:498, 0:20])
my_epochs_2 = rScaler.fit_transform(my_epochs[498:996, 0:20])
my_epochs_3 = rScaler.fit_transform(my_epochs[996:1494, 0:20])
my_epochs_4 = rScaler.fit_transform(my_epochs[1494:1992, 0:20])
my_epochs_5 = rScaler.fit_transform(my_epochs[1992:2329, 0:20])
my_epochs_6 = rScaler.fit_transform(my_epochs[2329:2748, 0:20])
my_epochs_7 = rScaler.fit_transform(my_epochs[2748:3246, 0:20])
my_epochs_8 = rScaler.fit_transform(my_epochs[3246:3744, 0:20])
my_epochs_9 = rScaler.fit_transform(my_epochs[3744:4242, 0:20])

summed_epochs = numpy.concatenate((my_epochs_1, my_epochs_2, my_epochs_3, my_epochs_4, my_epochs_5, my_epochs_6, my_epochs_7, my_epochs_8, my_epochs_9), axis=0)
summed_epochs = numpy.append(summed_epochs, my_epochs[:, -1].reshape(len(my_epochs[:, -1]), 1), axis=1)

# shuffle the training data
numpy.random.seed(2) 
numpy.random.shuffle(summed_epochs)
print(summed_epochs.shape)



# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(summed_epochs[0:index1, :], summed_epochs[0:index1, -1], random_state=1, test_size=0.3, shuffle = False)
print(X_train_tmp.shape)
print(X_test_tmp.shape)

# augment train data
choice = X_train_tmp[:, -1] == 0.
X_total = numpy.append(X_train_tmp, X_train_tmp[choice, :], axis=0)
#X_total_2 = numpy.append(X_total_1, X_train_tmp[choice, :], axis=0)
#X_total_3 = numpy.append(X_total_2, X_train_tmp[choice, :], axis=0)
#X_total_4 = numpy.append(X_total_3, X_train_tmp[choice, :], axis=0)
#X_total = numpy.append(X_total_1, X_train_tmp[choice, :], axis=0)
print(X_total.shape)

# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total, X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))
print(X_train_keep.shape)

train_data = numpy.append(X_train_keep, Y_train_keep.reshape(len(Y_train_keep), 1), axis=1)
numpy.random.shuffle(train_data)


#=======================================
 
# Data Pre-processing - scale data using robust scaler

Y_train = train_data[:, -1]
Y_test = X_test_tmp[:, -1]

input = train_data[:, 0:20]
testinput = X_test_tmp[:, 0:20]

#=====================================

# Model configuration

input = input.reshape(len(input), 1, 20)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 1, 20)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)

# Create the model
model=Sequential()
model.add(Conv1D(filters=22, kernel_size=4, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), padding='valid', activation='relu', strides=1, input_shape=(20, 1)))
model.add(Conv1D(filters=22, kernel_size=5, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), padding='valid', activation='relu', strides=1))
model.add(AveragePooling1D(pool_size=2))
model.add(Conv1D(filters=22, kernel_size=4, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), padding='valid', activation='relu', strides=1))
model.add((GlobalAveragePooling1D()))
model.add(Dense(60, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), activation='relu',))
model.add(Dense(2, activation='softmax'))

model.summary()

# Compile the model   
adam = Adam(learning_rate=0.000018)
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

hist = model.fit(input, Y_train, batch_size=32, epochs=70, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None, callbacks=[es, mc])

# evaluate the model
predict_y = model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.xlabel("No of iterations")
pyplot.ylabel("Accuracy and loss")
pyplot.show()

#==================================

model.save("model_conv1d.h5")

# load the saved model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(input, Y_train, verbose=1)
_, test_acc = saved_model.evaluate(testinput, Y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# evaluate the model
predict_y = saved_model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


#==================================

