import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.applications.efficientnet import preprocess_input


BASE_DATA_PATH = 'C:/Documents/Dataset/Original/TomatoPepper_Final_Split'
TEST_DIR = os.path.join(BASE_DATA_PATH, 'test')
WEIGHTS_PATH = 'best_EFFICIENTNET_model.h5' 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


print("\n Preparing Test Data...")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) # Essential for EfficientNet

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\n Rebuilding EfficientNet Model...")

base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(test_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


print(f" Loading Weights from {WEIGHTS_PATH}...")
model.load_weights(WEIGHTS_PATH)
print(" Weights Loaded Successfully!")


print(" Compiling model for testing...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


print("\n Running Final Test...")
results = model.evaluate(test_generator)
print(f"\n Final Test Accuracy: {results[1]*100:.2f}%")

print("\nGenerating Detailed Report...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
print(classification_report(test_generator.classes, y_pred, target_names=list(test_generator.class_indices.keys())))