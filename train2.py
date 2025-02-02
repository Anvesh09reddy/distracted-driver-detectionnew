from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the ResNet50 model pre-trained on ImageNet data
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Build the new model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),  # Add dropout for regularization
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # Assuming you have 10 classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Anvesh\Downloads\mini project\data\imgs\train',  # Absolute path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


# Train the model
model.fit(train_generator, epochs=10, steps_per_epoch=100)
# Unfreeze all layers of the base model
base_model.trainable = True

# Recompile the model (important for fine-tuning)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model
model.fit(train_generator, epochs=10, steps_per_epoch=100)
# Assuming you have a validation generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model
model.evaluate(val_generator)
