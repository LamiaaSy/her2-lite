import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Input,Softmax, Dense, Conv2D,Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from sklearn.metrics import confusion_matrix, classification_report


####################################################################################
#                           LOAD DATA
####################################################################################
# train_data_dir01 = 'dataset/train'
# valid_data_dir01 = 'dataset/validation'

####################################################################################
#                           PARAMETERS
####################################################################################
(img_width, img_height)=(224,224)
batch_size = 24
epochs = 80
n_classes=4

####################################################################################
#                           MODEL
####################################################################################

input_shape = (img_width, img_height, 3)
baseModel = EfficientNetV2B0(include_top=False, weights= 'imagenet' , input_shape=input_shape)
baseModel.trainable = True


############## NETWORK PRUNING
stop_layer = 'block4c_add'   # block5e_add  block4c_add  block3b_add  block2b_add  block1a_project_activation
topModel = Model(inputs=baseModel.input, outputs=baseModel.get_layer(stop_layer).output)
topModel = topModel.output
topModel= Conv2D(192, kernel_size=7,name='top_conv0')(topModel)

topModel = GlobalAveragePooling2D()(topModel)
topModel = BatchNormalization()(topModel)
topModel = Dense(1024, activation='relu')(topModel)

output_layer = Dense(4, activation='softmax')(topModel)

model = Model(inputs=baseModel.input, outputs=output_layer)

model.summary()


#####################################################################################################################################
#                                                            COMPLILE MODEL
#####################################################################################################################################


adam = keras.optimizers.Adam(learning_rate= 6e-3,amsgrad= True)

model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy','AUC','Precision','Recall'])


#####################################################################################################################################
#                                                            CHECK POINTS
#####################################################################################################################################

##LR SCHEDULER
reduce_lr = ReduceLROnPlateau(
             monitor='val_accuracy',
             factor=0.1,
             patience= 10,
             verbose=1,
             mode='max',
             min_lr=6e-7)

#### ModelCheckpoint
checkpoint01 = ModelCheckpoint(filepath='best_model.h5',
                               save_best_only=True,
                               monitor= 'val_accuracy',
                               mode= 'auto',
                               verbose=2)

# EarlyStopping
early_stop = EarlyStopping( monitor='val_accuracy',
                            patience=20,
                            restore_best_weights=True,
                            mode='auto')



#####################################################################################################################################
#                                              DATA GENERATORS
#####################################################################################################################################

##TRAIN
train_datagen = ImageDataGenerator( preprocessing_function=preprocess_input,
                                    horizontal_flip= True,
                                    vertical_flip=True,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05 )



train_generator01 = train_datagen.flow_from_directory(  train_data_dir01,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        shuffle= True,
                                                        class_mode='categorical')




##VALIDATION
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator01 =     valid_datagen.flow_from_directory(  valid_data_dir01,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            class_mode='categorical')


##TEST
#test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#test_generator =   test_datagen.flow_from_directory(
#                        test_data_dir01,
#                        target_size=(img_width, img_height),
#                        batch_size=batch_size,
#                        shuffle=False,
#                        class_mode='categorical')




#####################################################################################################################################
#                                   TRAIN MODEL
#####################################################################################################################################

print("[INFO] training model .............")

n_steps01 = train_generator01.samples // batch_size
n_val_steps01 = valid_generator01.samples // batch_size


model_history = model.fit(  train_generator01,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=valid_generator01,
                            steps_per_epoch=n_steps01,
                            validation_steps=n_val_steps01,
                            callbacks=[early_stop, reduce_lr, checkpoint01],
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
best_model = load_model('best_model.h5')

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

# #####################################################################################################################################
# #                                                                        METRICS
# #####################################################################################################################################

from sklearn.metrics import matthews_corrcoef


# Predict on the generator
y_pred_probs = model.predict(valid_generator01, verbose=1)

y_pred = y_pred_probs.argmax(axis=1)

y_true=valid_generator01.classes


mcc = matthews_corrcoef(y_true, y_pred)
print("MCC:", mcc)

