import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.activations import linear, relu, sigmoid
np.set_printoptions(edgeitems=2000, linewidth=2000)

#Load data
x_hold = np.load("data/X.npy")
y_hold = np.load("data/y.npy")

#split the data
x_train,x_,y_train,y_ = train_test_split(x_hold,y_hold,test_size=0.40,random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

print(x_train.shape)
print(y_train.shape)
print(x_cv.shape)
print(y_cv.shape)
print(x_test.shape)
print(y_test.shape)


#Displaying data
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

fig,axes = plt.subplots(3,3, figsize = (3,3))
fig.tight_layout(pad = 0.3)
for i,ax in enumerate(axes.flat):
    #picking a random training sample
    rand = np.random.randint(x_train.shape[0])

    #convert to its originial dimension
    converted = x_train[rand].reshape((20,20)).T

    #show the image
    ax.imshow(converted,cmap = "gray")

    #set the corresponding result
    ax.set_title(y_train[rand])
    ax.set_axis_off()

#plt.show()


#Creating model
tf.random.set_seed(1234) # for consistent result
model = Sequential(
    [
        tf.keras.Input(shape = (400,)),
        Dense(units = 25, activation="relu"),
        Dense(units = 15, activation="relu"),
        Dense(units = 10, activation="linear"),
    ]
)


#model summary
print(model.summary())

#compile model
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
)

#fit model
model.fit(x_train,y_train,epochs = 40)


#make a prediction
#prediction of 2
image_of_two = x_train[198]

prediction = model.predict(image_of_two.reshape(1,400))

print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

prediction_p = tf.nn.softmax(prediction)

print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")

yhat = np.argmax(prediction_p)
print(f"np.argmax(prediction_p): {yhat}")


#Displaying image with the corresponding actual and predicted value
fig,axes = plt.subplots(8,8,figsize = (8,8))
m = x_train.shape[0]
fig.tight_layout(pad = 0.1)
for i,ax in enumerate(axes.flat):
    rand = np.random.randint(m)

    #prediction
    pred = model.predict(x_train[rand].reshape(1,400))
    y_pred = tf.nn.softmax(pred)
    ypred_real = np.argmax(y_pred)

    conv = x_train[rand].reshape((20,20)).T

    ax.imshow(conv,cmap="gray")

    ax.set_title(f"{y_train[rand]},{ypred_real}")
    ax.set_axis_off()

plt.show()