import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load data and model
data = np.load('test_data.npz')
x_test, y_test = data['x_test'], data['y_test']
print(x_test.shape),print(y_test.shape)
model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print(x_test[10])
print("\n")
print(x_test[10].reshape(1,-1))

yhat_test = model.predict(x_test[10].reshape(1,-1))
y_pred_test = np.argmax(yhat_test, axis=1)
#y_pred_real = y_pred_test.reshape(-1,1)
train_classification_accuracy = accuracy_score([y_test[3]], y_pred_test)
print(train_classification_accuracy)
print(f"{y_test[10]},{y_pred_test}")

yhat_test = model.predict(x_test)
y_pred_test = np.argmax(yhat_test, axis=1)
#y_pred_real = y_pred_test.reshape(-1,1)
train_classification_accuracy = accuracy_score(y_test, y_pred_test)
print(train_classification_accuracy)
for i in range(len(y_test)):
    print(f"{y_test[i][0]},{y_pred_test[i]}")


