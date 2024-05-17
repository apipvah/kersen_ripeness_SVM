import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pathlib
from keras.src.optimizers import Adam
from tensorflow.python.keras.optimizers import adam_v2

# Set directory paths
train_dir = pathlib.Path('dataset/train')
val_dir = pathlib.Path('dataset/val')
test_dir = pathlib.Path('dataset/test')

# Parameters
BATCH_SIZE = 32
IMG_SIZE = (150, 150)

# Load datasets
train_dataset = tf.data.Dataset.list_files(str(train_dir/'*/*'), shuffle=True)
val_dataset = tf.data.Dataset.list_files(str(val_dir/'*/*'), shuffle=True)
test_dataset = tf.data.Dataset.list_files(str(test_dir/'*/*'), shuffle=True)

# Function to process paths and load images
def process_path(file_path):
    parts = tf.strings.split(file_path, '/')
    label = tf.strings.split(parts[-2], '_')[-1]
    label = tf.where(label == 'matang', 1, 0)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

# Process datasets
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Prepare batches
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Predict on a new image
def predict_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        return "Matang"
    else:
        return "Belum Matang"

# Contoh penggunaan
img_path = '../dataset/val/mentah_1.jpeg'
result = predict_image(img_path)
print(f'Buah kersen tersebut {result}.')
