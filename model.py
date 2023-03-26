import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
tf.keras.datasets.mn

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D())
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax')) # must be 10 since 0-9 number classes

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=x_train,y=y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)

model.save('simplified_model')
print('\nTest accuracy:', test_acc)
model.save('model.h5')