from tensorflow.keras.layers import LeakyReLU,BatchNormalization,Dense,Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,load_model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from sklearn.svm import SVC
from glob import glob
import numpy as np
import os
import cv2
#%%
#############load images and labels ##############
print("[INFO] loading images...")
imagePaths ='upwork/data/*/*.jpg'
data = []
labels = []
# loop over the image paths
for imagePath in glob(imagePaths):
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (64x64) and preprocess it
    image = cv2.imread(imagePath,0)
    image=cv2.resize(image,(64,64))
    image=image/255.0
    flatten=np.reshape(image,(-1))
    # update the data and labels lists, respectively
    data.append(flatten)
    labels.append(label)
# convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)
#convert labels into a one 1d vector
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels=np.reshape(labels,(-1))

# train_test_split
X_train, X_test, y_train, y_test= train_test_split(data, labels,test_size=0.20,
                                                   stratify=labels, random_state=42)
#%%
#############take a look at the images##############
import matplotlib.pyplot as plt
def up(im):
    return np.reshape(im,(64,64))
def display(reference0, out2, rotated, dst):
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(reference0,cmap="gray")
    axarr[0, 0].title.set_text('labels 1')
    axarr[0, 1].imshow(out2,cmap="gray")
    axarr[0, 1].title.set_text('labels 0')
    axarr[1, 0].imshow(rotated,cmap="gray")
    axarr[1, 0].title.set_text('labels 0')
    axarr[1, 1].imshow(dst,cmap="gray")
    axarr[1, 1].title.set_text('labels 1')
    plt.show()
ind=23
display(up(X_train[ind]),up(X_train[ind+1]),up(X_train[ind+2]),up(X_train[ind+3]))
print(y_train[ind],y_train[ind+1],y_train[ind+2],y_train[ind+3])
# cv2.imshow("X_train[1]",np.reshape(X_train[6],(64,64)))
# cv2.waitKey(0)
#%%
#############the network##############
n_inputs = X_train.shape[1]
# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 3
e = Dense(round(float(n_inputs) / 2.0))(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck n_inputs/4
n_bottleneck = round(float(n_inputs) / 4.0)
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(n_inputs*2)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 3
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
#%%
#############train the network a plot the loss/epochs##############
# plot the autoencoder
plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,X_test))
# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
#%%
#############split the encoder and save it##############
# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
# save the encoder to file
encoder.save('AutoencoderFeatureExtraction/encoder.h5')
#%%
encoder = load_model('AutoencoderFeatureExtraction/encoder.h5')

#%%
#############extract features and train svm classifier##############
X_train_features=encoder.predict(X_train)
X_test_features=encoder.predict(X_test)

# ["linear","rbf","poly","sigmoid"]
classifier = SVC(kernel='linear')
# classifier = SVC(kernel='rbf', random_state = 1)
# classifier = SVC(kernel='poly', degree=8)
# classifier = SVC(kernel='sigmoid')
classifier.fit(X_train_features, y_train)

Y_pred = classifier.predict(X_test_features)
cm = confusion_matrix(y_test, Y_pred)
accuracy = float(cm.diagonal().sum()) / len(y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)
