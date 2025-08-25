import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten

'''
count = 0
dirs = os.listdir('Images/train/')
for dir in dirs:
    files = list(os.listdir('Images/train/'+dir))
    print(dir + ' Image Folder has ' + str(len(files)) + ' Images')
    count = count + len(files)
print('train folder has ' + str(count) + ' Images')
'''

base_dir = 'Images/train'
img_size = 180
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(base_dir,
                                                        seed = 123,
                                                        validation_split=0.1,
                                                        subset="training",
                                                        batch_size=batch_size,
                                                        image_size=(img_size,img_size))

val_ds = tf.keras.utils.image_dataset_from_directory(base_dir,
                                                        seed = 123,
                                                        validation_split=0.1,
                                                        subset="validation",
                                                        batch_size=batch_size,
                                                        image_size=(img_size,img_size))

flower_names = train_ds.class_names
'''
print(flower_names)
'''

'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

for images,labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(flower_names[labels[i]])
        plt.axis('off')
'''

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

# Data Augmentation

data_augmentation = Sequential([
    layers.RandomFlip("horizontal",input_shape = (img_size,img_size,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Model Creation
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    Conv2D(16,3,padding="same",activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding="same", activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding="same", activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(5)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''
print(model.summary)
'''
history = model.fit(train_ds,epochs=30,validation_data=val_ds)

input_image = tf.keras.utils.load_img('Images/test/Image_1.jpg',target_size=(180,180))
input_image_array = tf.keras.utils.img_to_array(input_image)
input_image_exp_dim = tf.expand_dims(input_image_array,0)

predictions = model.predict(input_image_exp_dim)
result = tf.nn.softmax(predictions[0])
flower_name = flower_names[np.argmax(result)]
print(flower_name)

def classify_images(image_path):
    input_image=tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(max(result))
    return outcome


model.save('Flower_Recog_Model.h5')
