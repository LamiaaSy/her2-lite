import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, LeakyReLU, Input
from keras.layers.merging import concatenate
from keras import optimizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import CBAM_DEFS as cbam

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


####################################################################################
#                           LOAD DATA
####################################################################################

# train_data_dir01 = 'dataset/train'
# valid_data_dir01 = 'dataset/train/validation'

####################################################################################
#                           PARAMETERS
####################################################################################

img_width01, img_height01 = 512, 512

num_classes = 2
batch_size = 24
epochs_pretrain = 1


###################################################################################################################################
#                            MODEL
###################################################################################################################################


inp = Input(shape=(img_width01, img_height01, 3))
convCOM = Conv2D(16, kernel_size=3, strides=1,padding='same',activation=LeakyReLU(0.001))(inp)
poolCOM = MaxPooling2D(pool_size=2, strides=2)(convCOM)


###########################LEFT BRANCH

convLEFT1 = Conv2D(64, kernel_size=3, strides=3,padding='same', kernel_regularizer=l2(1e-5),activation=LeakyReLU(0.001))(poolCOM)
poolLEFT1 = MaxPooling2D(pool_size=3, strides=2)(convLEFT1)
dropLEFT1 = Dropout(0.1)(poolLEFT1)
attLEFT1= cbam.cbam_block(dropLEFT1,features=64, kernel=3, spatial=True,name='attLEFT1')

convLEFT2 = Conv2D(128, kernel_size=3, strides=3, padding='same',kernel_regularizer=l2(1e-5))(attLEFT1)
actvLEFT2 = LeakyReLU(0.001)(convLEFT2)
poolLEFT2 = MaxPooling2D(pool_size=3, strides=2)(actvLEFT2)
dropLEFT2 = Dropout(0.1)(poolLEFT2)

flatLEFT = GlobalAveragePooling2D()(dropLEFT2)


#########################RIGHT  BRANCH
convRIGHT1a = Conv2D(32, kernel_size=3, strides=1, padding = 'same', kernel_regularizer=l2(0.00001), activation=LeakyReLU(0.001) )(poolCOM)
poolRIGHT1a = MaxPooling2D(pool_size=3, strides=1)(convRIGHT1a)

convRIGHT1b = Conv2D(64, kernel_size=3, strides=2, kernel_regularizer=l2(0.00001), padding='same', activation=LeakyReLU(0.001))(poolRIGHT1a)
poolRIGHT1b = MaxPooling2D(pool_size=3, strides=2)(convRIGHT1b)
batchRIGHT1 = BatchNormalization()(poolRIGHT1b)

convRIGHT2a = Conv2D(128, kernel_size=5, strides=2, kernel_regularizer=l2(0.00001), padding='same', activation=LeakyReLU(0.001))( batchRIGHT1)
poolRIGHT2  = MaxPooling2D(pool_size=3, strides=3)(convRIGHT2a)

flatRIGHT = GlobalAveragePooling2D()(poolRIGHT2)

####################CONCATENATE
merge = concatenate([flatLEFT, flatRIGHT])

hidden1 = Dense(128, activation=LeakyReLU(0.001))(merge)  ##orignally 96 ## try samLLLER ?

output = Dense(1, activation='sigmoid')(hidden1)

model = Model(inputs=inp, outputs=output)

print(model.summary())

#####################################################################################################################################
#                                                            COMPLILE MODEL
#####################################################################################################################################

adam01 = optimizers.Adam(learning_rate=6e-2, epsilon=1, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=adam01, metrics=['accuracy', 'AUC','Precision','Recall'])


#####################################################################################################################################
#                                                            CHECK POINTS
#####################################################################################################################################

##LR SCHEDULER
reduce_lr = ReduceLROnPlateau(
             monitor='val_accuracy',
             factor=0.1,
             patience=6,
             verbose=1,
             mode='max',
             min_lr=6e-7)


early_stop = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience = 12,
    verbose=0,
    mode='max')


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_ather2.h5',
    monitor='val_accuracy',
    mode='max',
    verbose=0,
    save_best_only=True,
    )


#####################################################################################################################################
#                                                        PREPROCESSING
#####################################################################################################################################

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


#####################################################################################################################################
#                                              DATA GENERATORS
#####################################################################################################################################

train_datagenAUG = ImageDataGenerator(rescale=1. / 1,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      preprocessing_function=preprocess_input)

train_datagen = ImageDataGenerator(rescale=1. / 1,
                                   preprocessing_function=preprocess_input)

train_generator01 = train_datagen.flow_from_directory(train_data_dir01,
                                                      target_size=(img_width01, img_height01),
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      class_mode='binary')

##validation
valid_datagen = ImageDataGenerator(rescale=1. / 1, preprocessing_function=preprocess_input)

valid_generator01 = valid_datagen.flow_from_directory(valid_data_dir01,
                                                      target_size=(img_width01, img_height01),
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      class_mode='binary')


##test
test_datagen = ImageDataGenerator(rescale=1. / 1, preprocessing_function=preprocess_input)

test_generator01 = test_datagen.flow_from_directory(valid_data_dir01,
                                                    target_size=(img_width01, img_height01),
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    class_mode='binary')

#####################################################################################################################################
#                                   TRAIN MODEL
#####################################################################################################################################

print("[INFO] training model .............")

train_steps = math.ceil(train_generator01.samples / batch_size)
valid_steps = math.ceil(valid_generator01.samples / batch_size)

model_history = model.fit(train_generator01,
                          batch_size=batch_size,
                          epochs=epochs_pretrain,
                          validation_data=valid_generator01,
                          steps_per_epoch=train_steps,
                          validation_steps=valid_steps,
                          callbacks=[model_checkpoint,early_stop,reduce_lr ],
                          verbose=2)


#####################################################################################################################################
#                              PLOT CURVES
#####################################################################################################################################

hist = model.history.history['val_accuracy']
n_epochs_best = np.argmax(hist)

print("[INFO] plotting loss and accuracy...")

loss =model_history.history["loss"]
val_loss =  model_history.history["val_loss"]
acc= model_history.history["accuracy"]
val_acc=model_history.history["val_accuracy"]

N=range(1, len(loss) + 1)
print('>>Best Epoch: ',n_epochs_best+1)
print('>>Stopped Epoch: ', N)

plt.style.use("ggplot")
plt.figure(1)
plt.plot( np.arange(1,len(loss)+1), loss , label="train_loss",color="blue")
plt.plot(np.arange(1,len(loss)+1),val_loss, label="val_loss", color="orange")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

plt.figure(2)
plt.plot(np.arange(1,len(loss)+1),acc, label="train_acc",color="blue")
plt.plot(np.arange(1,len(loss)+1),val_acc, label="val_acc",color="orange")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")



###Confusion Matrix
best_model = load_model('best_ather2.h5')

y_probs = best_model.predict(valid_generator01)
y_pred = np.argmax(y_probs, axis=1)
y_true = valid_generator01.classes

cm2 = confusion_matrix(y_true, y_pred)

labels = ['0', '1+', '2+', '3+']
print(classification_report(y_true, y_pred, target_names=labels))

plt.figure(figsize=(6, 5))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

