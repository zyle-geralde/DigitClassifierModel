import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from PIL import Image

#getting image
image_path = f'testImages/image_{945}.png'  # Replace with your image path
image = Image.open(image_path)

# Convert to grayscale (if needed)
image = image.convert('L')  # 'L' mode is for grayscale

# Convert the image to a NumPy array
image_array = np.array(image)

# Normalize pixel values to the range [0, 1]
# Pixel values range from 0 to 255 in grayscale images
normalized_array = image_array / 255.0
normalized_array = normalized_array.T
normalized_array = normalized_array.reshape((1, 400))


# Load data and model
data = np.load('test_data.npz')
x_test, y_test = data['x_test'], data['y_test']
print(x_test.shape),print(y_test.shape)
model = tf.keras.models.load_model('best_model.keras')

yhat_test = model.predict(normalized_array)
y_pred_test = np.argmax(yhat_test, axis=1)
#y_pred_real = y_pred_test.reshape(-1,1)
print(y_pred_test[0])


'''EXTRA NOTES'''
#converting images to 1X400 matrix to be used in processing
'''array_x = np.zeros((1000,400))
for i in range(1000):
    # Load the image
    image_path = f'testImages/image_{i}.png'  # Replace with your image path
    image = Image.open(image_path)

    # Convert to grayscale (if needed)
    image = image.convert('L')  # 'L' mode is for grayscale

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize pixel values to the range [0, 1]
    # Pixel values range from 0 to 255 in grayscale images
    normalized_array = image_array / 255.0
    normalized_array = normalized_array.T
    normalized_array = normalized_array.reshape((1,400))

    array_x[i] = normalized_array'''

# Evaluate the model
#test_loss, test_accuracy = model.evaluate(normalized_array, y_test[945,:])
#print(f"Test Accuracy: {test_accuracy:.4f}")


# Save images from its grayscale encoded to an actual image locally
'''for i in range(len(x_test)):
    # Normalize and clip data
    image_data = x_test[i].copy()
    image_data = np.clip(image_data, 0, 1)  # Clip values to [0, 1]
    image_data = (image_data * 255).astype(np.uint8)  # Scale to [0, 255]

    # Ensure correct dimensions (e.g., 20x20 for grayscale)
    image_data = image_data.reshape(20, 20).T  # Adjust dimensions as needed

    # Create a PIL Image from the NumPy array
    image = Image.fromarray(image_data, mode='L')  # 'L' mode for grayscale

    # Save the image
    image.save(f'testImages/image_{i}.png')
    print(f"Saved image_{i}.png")'''




