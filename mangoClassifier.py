import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# Set seeds for reproducibility
seed = 7
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load the dataset from Excel
data = pd.read_excel('C:\\Users\\User\\Desktop\\DL\\tree_test_1\\tree_species_data.xlsx')

# Display the first few rows of the dataset
print(data.head())
print(data.shape)  # Output: (120, 4)
##########################################################################################
# Split the dataset into features (X) and labels (y)
X = data.iloc[:, 1:].values  # Features: columns 2 to end
y = data.iloc[:, 0].values   # Labels: column 1

# Encode labels to integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# Convert integers to one-hot encoding
one_hot_y = np_utils.to_categorical(encoded_y)

# Split the dataset into training (80%) and testing (20%) sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, one_hot_y, test_size=0.20, random_state=seed)

# Further split the training data into training (80% of 80%) and validation (20% of 80%) sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=seed)

# Normalize the features (This will improve training)
X_train = X_train / np.max(X_train)
X_val = X_val / np.max(X_val)
X_test = X_test / np.max(X_test)
##########################################################################################
# Build and Train the neural network model
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, input_shape=(X_train.shape[1],)),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=20)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
##########################################################################################
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
##########################################################################################
# Predict labels for the test data
test_labels_predicted = np.argmax(model.predict(X_test), axis=-1)

# Compute confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), test_labels_predicted)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap='Greens')  
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate and print test accuracy
test_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print("Test Accuracy (calculated from confusion matrix):", test_accuracy)
##########################################################################################
# K-Fold Cross Validation
from sklearn.model_selection import train_test_split, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
for train_index, val_index in kf.split(X_train_full):
    X_train_k, X_val_k = X_train_full[train_index], X_train_full[val_index]
    y_train_k, y_val_k = y_train_full[train_index], y_train_full[val_index]

    print(f'Training for fold {fold_no} ...')

    # Train the model
    history = model.fit(X_train_k, y_train_k, validation_data=(X_val_k, y_val_k), epochs=10, batch_size=32)

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val_k, y_val_k)
    print(f"Validation Loss for fold {fold_no}:", val_loss)
    print(f"Validation Accuracy for fold {fold_no}:", val_accuracy)

    fold_no += 1
        
#save model
model.save('mango_classifier_01.h5')

# Function to predict the type of mango tree
def predict_mango_tree(ratio, angle, length_of_petiole):
    # Assuming 'max_X_train' is the maximum value from X_train used for normalization during training
    max_X_train = np.max(X_train_full) 
        
    # Prepare the input data by normalizing it the same way as the training data
    input_data = np.array([[ratio, angle, length_of_petiole]]) / max_X_train
    
    # Predict the type of mango tree
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction, axis=1)
    
    # Decode the prediction to get the mango tree type
    predicted_tree = encoder.classes_[predicted_class_index][0]  # Access class name directly
    
    return predicted_tree

# Example usage
example_prediction = predict_mango_tree(3.5, 28.2, 1.78)
print(f'Predicted Mango Tree: {example_prediction}')
