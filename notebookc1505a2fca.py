# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:37:40.141195Z","iopub.execute_input":"2025-02-19T02:37:40.141592Z","iopub.status.idle":"2025-02-19T02:37:56.138589Z","shell.execute_reply.started":"2025-02-19T02:37:40.141565Z","shell.execute_reply":"2025-02-19T02:37:56.137342Z"}}
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Flatten, Dense,Dropout, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:38:03.934129Z","iopub.execute_input":"2025-02-19T02:38:03.934463Z","iopub.status.idle":"2025-02-19T02:38:03.938848Z","shell.execute_reply.started":"2025-02-19T02:38:03.934399Z","shell.execute_reply":"2025-02-19T02:38:03.937534Z"}}

BASE_DIR="/kaggle/input/melanoma-cancer-dataset"
TRAIN_DIR=os.path.join(BASE_DIR,"train")
TEST_DIR=os.path.join(BASE_DIR,"test")

# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:38:12.210028Z","iopub.execute_input":"2025-02-19T02:38:12.210452Z","iopub.status.idle":"2025-02-19T02:38:12.214907Z","shell.execute_reply.started":"2025-02-19T02:38:12.210390Z","shell.execute_reply":"2025-02-19T02:38:12.213586Z"}}
IMG_SIZE=(224,224)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:38:18.461929Z","iopub.execute_input":"2025-02-19T02:38:18.462262Z","iopub.status.idle":"2025-02-19T02:38:18.466726Z","shell.execute_reply.started":"2025-02-19T02:38:18.462235Z","shell.execute_reply":"2025-02-19T02:38:18.465217Z"}}
BATCH_SIZE=32

# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:38:44.127376Z","iopub.execute_input":"2025-02-19T02:38:44.127795Z","iopub.status.idle":"2025-02-19T02:38:53.889350Z","shell.execute_reply.started":"2025-02-19T02:38:44.127764Z","shell.execute_reply":"2025-02-19T02:38:53.888317Z"}}
train_data_generator=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    
)

test_data_generator=ImageDataGenerator(rescale=1./255)
train_generator=train_data_generator.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
                                                        )
test_generator = test_data_generator.flow_from_directory(
    directory=TEST_DIR, 
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)





# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:38:56.594106Z","iopub.execute_input":"2025-02-19T02:38:56.594475Z","iopub.status.idle":"2025-02-19T02:38:57.006614Z","shell.execute_reply.started":"2025-02-19T02:38:56.594410Z","shell.execute_reply":"2025-02-19T02:38:57.005564Z"}}
from tensorflow.keras import layers, models

# Build the CNN model
model = models.Sequential([
    # 1st Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # 2nd Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 3rd Convolution Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening Layer
    layers.Flatten(),
    
    # Fully Connected Layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',  # Optimizer for faster convergence
              loss='binary_crossentropy',  # Binary classification loss
              metrics=['accuracy'])  # Accuracy metric

# Model Summary
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2025-02-18T19:50:37.956623Z","iopub.execute_input":"2025-02-18T19:50:37.957051Z","iopub.status.idle":"2025-02-18T21:59:33.093205Z","shell.execute_reply.started":"2025-02-18T19:50:37.956985Z","shell.execute_reply":"2025-02-18T21:59:33.092158Z"}}





early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,  # Stop training after 5 epochs with no improvement
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau to reduce the learning rate if the model stops improving
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,  # Reduce LR if no improvement after 3 epochs
    min_lr=1e-7,
    verbose=1
)
model.fit(train_generator,steps_per_epoch=train_generator.samples//BATCH_SIZE,
         epochs=30,
         validation_data=test_generator,
         validation_steps=test_generator.samples//BATCH_SIZE,
        callbacks=[early_stopping, lr_scheduler]

         )


# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:40:35.570291Z","iopub.execute_input":"2025-02-19T02:40:35.570645Z","iopub.status.idle":"2025-02-19T02:40:37.107843Z","shell.execute_reply.started":"2025-02-19T02:40:35.570623Z","shell.execute_reply":"2025-02-19T02:40:37.106652Z"}}
# Save Model
model.save("skin_cancer_cnn.h5")
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the entire model
model = load_model('/kaggle/working/skin_cancer_cnn.h5')


def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"
    
    # Show Image with Prediction
    plt.imshow(img)
    plt.title(f"Predicted: {class_label}")
    plt.axis("off")
    plt.show()
# Test on new image
predict_skin_cancer("/kaggle/input/melanoma-cancer-dataset/test/Malignant/5616.jpg", model)


# %% [code] {"execution":{"iopub.status.busy":"2025-02-19T02:56:44.977887Z","iopub.execute_input":"2025-02-19T02:56:44.978231Z","iopub.status.idle":"2025-02-19T02:56:45.187392Z","shell.execute_reply.started":"2025-02-19T02:56:44.978202Z","shell.execute_reply":"2025-02-19T02:56:45.186346Z"}}
predict_skin_cancer("/kaggle/input/melanoma-cancer-dataset/train/Malignant/1004.jpg", model)
