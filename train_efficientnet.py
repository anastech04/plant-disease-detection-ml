import tensorflow as tf
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

BASE_DATA_PATH = 'C:/Documents/Dataset/Original/TomatoPepper_Final_Split'
TRAIN_DIR = os.path.join(BASE_DATA_PATH, 'train')
VALIDATION_DIR = os.path.join(BASE_DATA_PATH, 'val')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16 #Reduced to 16 because EfficientNet uses more VRAM and we are doing the training locally
EPOCHS = 20
LEARNING_RATE = 1e-4 

from tensorflow.keras.applications.efficientnet import preprocess_input

print("\n Initializing EfficientNet Generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f" Classes Detected: {NUM_CLASSES}")

print("\n Downloading & Building EfficientNetB0...")

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the top 20 layers for fine-tuning immediately
base_model.trainable = True
# Freeze the bottom layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x) 
x = Dropout(0.4)(x)         
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_EFFICIENTNET_model.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
]

print("\n Starting EfficientNet Training on Augmented Data...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n Training Finished")