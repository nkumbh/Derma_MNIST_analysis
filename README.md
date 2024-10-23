# Derma_MNIST_analysis


# Download the DermaMNIST dataset
!pip install medmnist
from medmnist import DermaMNIST
train_dataset = DermaMNIST (split='train', download=True)
train_dataset
# *Import Libraries*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive

# Load dataset
!wget -O Data.npz https://zenodo.org/record/4269852/files/dermamnist.npz?download=1
data = np.load('Data.npz')
print(data.files)
print(data.files)
print(data['train_images'])
# Analyse the Data
print(f'Train Set:      X:%s   Y:%s' %(data['train_images'].shape, data['train_labels'].shape))
print(f'Validation Set: X:%s   Y:%s' %(data['val_images'].shape, data['val_labels'].shape))
print(f'Test Set:       X:%s   Y:%s' %(data['test_images'].shape, data['test_labels'].shape))
# Data concatenation
* X axis contains *Images* and y axis contains *Labels*
X_train = data['train_images']
X_val = data['val_images']
X_test = data['test_images']
X = np.concatenate((X_train, X_val, X_test), axis=0)
y_train = data['train_labels']
y_val = data['val_labels']
y_test = data['test_labels']
y = np.concatenate((y_train, y_val, y_test), axis=0)
X.shape, y.shape
//*Create a disctionary with Labels and its respective pigmented skin lesions type*
y_val.shape
y_test.shape
labels = [0, 1, 2, 3, 4, 5, 6]
lesions_type_dict = {'0': 'Actinic keratoses','1': 'Basal cell carcinoma','2': 'Benign keratosis-like lesions ',
                     '3': 'Dermatofibroma','4': 'Melanocytic nevi','5': 'Vascular lesions','6': 'Melanoma'}
lesions_type_dict
* _number of samples of each class in the whole dataset_
num_classes = []
for i in range(len(labels)):
    num_classes.append(len(np.where(y==i)[0]))
pd.DataFrame(num_classes, index=labels)
y_test.shape
# Data Visualization
fig, ax = plt.subplots(7, 10)
fig.set_figheight(10)
fig.set_figwidth(10)
for classes in range (7):
  for i, inx in enumerate(np.where(y==classes)[0][:10]):
    ax[classes,i].imshow(X[inx])
    ax[classes,i].set_ylabel(labels[classes],fontsize = 20.0)
    ax[classes,i].label_outer()
# __Data Preprocessing and Augmentation__

__Resampling__
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Initial class distribution
print("Initial class distribution:", Counter(y.flatten()))

# Define the over sampling strategy

oversampling_strategy = {0: num_classes[0]*8,
           1: num_classes[1]*5,
           2: 2500,
           3: 2500,
           4: 2500,
           5: num_classes[5],
           6: 2500}
smote = SMOTE(sampling_strategy=oversampling_strategy )  # Balance classes by generating minority class samples

oversampled_X , oversampled_y = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
print("Class distribution after under sampling:", oversampled_X.shape)
print("Class distribution after under sampling:", oversampled_y.shape)

y_test.shape
# Define the resampling strategy
undersample = RandomUnderSampler(sampling_strategy={0: 1500,
           1: 1500,
           2: 1500,
           3: 1500,
           4: 1500,
           5: 1500,
           6: 1500
})
# Apply the resampling pipeline to the data
undersampled_X, undersampled_y = undersample.fit_resample(oversampled_X, oversampled_y)

# Display the class distribution after resampling
print("Class distribution after under sampling:", undersampled_X.shape)
print("Class distribution after under sampling:", undersampled_y.shape)
y_test.shape
undersampled_X = undersampled_X.reshape(-1,28,28,3)
num_classes = []
for i in range(len(labels)):
  num_classes.append(len(np.where(undersampled_y==i)[0]))

pd.DataFrame(num_classes,index=labels)
y_val.shape
from tensorflow.keras.utils import to_categorical
undersampled_y = to_categorical(undersampled_y)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
y_val.shape
y_test.shape
# Image Data Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                rotation_range = 20,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True,
                                vertical_flip = True,
                                fill_mode = 'wrap')
batch_size = 40
train_data = train_datagen.flow(undersampled_X, undersampled_y, batch_size = batch_size, seed=1)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_data = test_datagen.flow(X_val, y_val, batch_size=batch_size,seed=1)
print(f'Train Set:      X:%s Y:%s' %(train_data.x.shape, train_data.y.shape))
print(f'Validation Set: X:%s Y:%s' %(val_data.x.shape, val_data.y.shape))
print(f'Test Set :      X:%s Y:%s' %(X_test.shape, y_test.shape))
# Model Creation
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
input_layer = Input(shape=(28,28,3))

# convolution block 1
cb11 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(input_layer)
cb12 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(cb11)
maxpl1 = MaxPool2D((2,2))(cb12)
#convolution block 2
cb21 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(maxpl1)
cb22 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(cb21)
maxpl2 = MaxPool2D((2,2))(cb22)
#convolution block 3
cb31 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(maxpl2)
cb32 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(cb31)
maxpl3 = MaxPool2D((2,2))(cb32)
#convolution block 4
cb41 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(maxpl3)
cb42 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(cb41)
cb43 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(cb42)
maxpl4 = MaxPool2D((2,2))(cb43)
# artificial neural network block
flat   = Flatten()(maxpl4)
dense1 = Dense(1024, activation="relu")(flat)
dense2 = Dense(1024, activation="relu")(dense1)
dense3 = Dense(1024, activation="relu")(dense2)
output = Dense(7, activation="softmax")(dense3)
model = Model(inputs=input_layer, outputs=output)
from keras.optimizers import Adam
import tensorflow as tf

# Define the learning rate and decay
learning_rate = 0.0001
decay_rate = 1e-5

# Initialize Adam optimizer with decay
adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay=decay_rate)

# Compile the model with the Adam optimizer
model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])
batch_size = 50
epochs = 40
model_history = model.fit(train_data,
                          steps_per_epoch= int(train_data.n/batch_size),
                          epochs=epochs,
                          validation_data=val_data,
                          validation_steps=int(val_data.n/batch_size))
model_history.params
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model)
model.evaluate(X_test/255, y_test)
pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()
y_proba = model.predict(X_test/255)
y_proba.round(2)
y_pred = np.argmax(y_proba, axis=-1)
y_pred[:10]
y_pred_name = np.array(labels)[y_pred]
y_pred_name[:10]
plt.figure(figsize=(20,10))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i])
    plt.title('True label: {}, Prediction: {}'.format(labels[y_test[i].argmax()], y_pred_name[i]))
plt.tight_layout()
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

y_test_numbers = np.array([y.argmax() for y in y_test])

cm = confusion_matrix(y_test_numbers, y_pred)

cm_display = ConfusionMatrixDisplay(cm,display_labels=labels)
fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
# Visualizing filters and output of the first layer of model
model.layers
first_conv = model.layers[1]
print(first_conv)
print(first_conv.weights)
print('Output Shape of the first Convolution layer: ',first_conv(X_test/255).shape)
print('Number of Testset data:                      ',first_conv(X_test/255).shape[0])
print('Number of first Convolution layer:           ',first_conv(X_test/255).shape[3])
n = 0
plt.imshow(X_test[n])
#### ploting 9 first filter
#first layer output
current_layer_output = first_conv(X_test/255)[n]
print(current_layer_output.shape)
fig, ax = plt.subplots(3, 3)
for i in range(9):
  ax[i//3][i%3].imshow(current_layer_output[:,:,i])
  ax[i//3][i%3].axis('off')
plt.tight_layout()
