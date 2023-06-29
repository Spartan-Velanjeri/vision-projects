import tensorflow as tf

from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.models import load_model


# We are loading the datasets directly from Keras's toy datasets
# that can be used for debugging or testing a model

# We use the CIFAR10 Dataset which has 6000 images for each of the 10
# classes. It has already been divided into 50k training and 10k testing images

(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()

# Normalise pixel values to be between 0 and 1. This is done to avoid
# computation of high numeric values which would become complex otherwise

train_images,test_images = train_images / 255.0, test_images / 255.0



# Splitting the training dataset into training and validation sets

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Let's verify the data. Let's plot 15 images from the training set and display their class names
# below

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, hence you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#Let's build the model!

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(32,32,3))) #the first argument is the number of filters (32)
# Output_size = ((input_size - kernel_size + 2*padding)/stride)+1
# default value of padding is valid (0 padding) and stride = (1,1)
model.add(layers.MaxPooling2D((2,2)))

# The pooling operation reduces the spatial dimensions of the 
# input while preserving important features.
# Number of param = (Kernel_size * Kernel_size * Input channels + 1) * (output_channels == no. of filters)

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))


# model.summary()

model.add(layers.Flatten()) # Dense layers take vectors (1D) as input, hence we flatten
model.add(layers.Dense(64,activation='relu')) #cuz last layer is 64
model.add(layers.Dense(10)) # because 10 classes

model.summary()

#Compile and train the model

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images,train_labels,epochs=10,
                    validation_data=(val_images,val_labels))

# Training fo 10 epochs

# Evaluate the model
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)

#Save your model in a h5 format, makes it easy with working with other applications
#model.save('simple_model.h5')

# Load the model from the .h5 file (Instead of training again)
# model = load_model('simple_model.h5')

plt.figure(figsize=(10,10))

# Make predictions for the test images

for i in range(15):
    test_image = np.expand_dims(test_images[i],axis=0)
    predictions = model.predict(test_image)

    # Get the predicted class labels
    predicted_labels = np.argmax(predictions, axis=1)

    ground_truth = test_labels[i]
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    # Set the title to include both labels
    title = f'Predicted: {class_names[predicted_labels[0]]}\nGround Truth: {class_names[ground_truth[0]]}'
    plt.title(title, fontsize=10, color='black', pad=6)
plt.show()

    
