---
name: MNIST_digits dataset
tools: [numpy, pandas, pytjon, matplotlib, collab]
image: /assets/img/MNIST_digits/output_2_0.png
description: Exploring MNIST_digits dataset 
---


```python
# Importing useful libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, BatchNormalization, Dropout
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout

```


```python
#Load the MNIST_digits dataset
mnist_digits = keras.datasets.mnist

# Split the dataset into training and testing sets
(X_train_full, y_train_full), (X_test, y_test) = mnist_digits.load_data()
```


```python
# Display the selected image
index_to_display = 0
plt.imshow(X_train_full[index_to_display], cmap='gray')  # 'cmap' specifies the color map (grayscale)
plt.title(f"Label: {y_train_full[index_to_display]}")    # Display the corresponding label as the title
plt.show()
```


    
![png](/assets/img/MNIST_digits/output_2_0.png)
    



```python
# Accessing the first data point in the training dataset X_train_full
X_train_full[0]
```




    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
             18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
            253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
            253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
            253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
            205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
             90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
            190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
            253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
            241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
            148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
            253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
            253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
            195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
             11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0]], dtype=uint8)




```python
X_train_full.shape
```




    (60000, 28, 28)




```python
y_train_full.shape
```




    (60000,)




```python
# Create a figure with a specified size (20x14)
plt.figure(figsize=(20, 14))

# Loop through the first 5 images in the training dataset
for index in range(5):
    # Create a subplot in a 1x5 grid, with the index + 1 specifying the position
    plt.subplot(1, 5, index + 1)

    # Display the image from the training dataset in binary (black and white)
    plt.imshow(X_train_full[index], cmap='binary')

    # Turn off axis labels and ticks for a cleaner appearance
    plt.axis('off')

    # Set the title of the subplot to display the target label (y_train_full) for this image
    # Customize the title font size to 20 and color to grey
    plt.title(f'Target: {y_train_full[index]}', fontsize=20, c='grey')

# Show the entire plot with the subplots
plt.show()

```


    
![png](/assets/img/MNIST_digits/output_6_0.png)
    



```python
Counter(y_train_full)
```




    Counter({5: 5421,
             0: 5923,
             4: 5842,
             1: 6742,
             9: 5949,
             2: 5958,
             3: 6131,
             6: 5918,
             7: 6265,
             8: 5851})



Training a NN with Keras
1. Preprocess data
2. Create model (NN architecture)
3. Compile (specify loss function, optimizers, metrics)
4. Fit (training & validate, batch_size, \# epochs)
5. Evaluate


```python
# 1. Preprocess data (both training and testing datasets)

# Scale to [0,1]
X_train_full = X_train_full/255
X_test = X_test/255
```


```python
# Flatten the data
X_train_full = X_train_full.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
```


```python
# Convert labels to one-hot encoding
#y_train_full = to_categorical(y_train_full, 10)
#y_test = to_categorical(y_test, 10)
```


```python
# Create validation dataset
from sklearn.model_selection import train_test_split
X_train_tr, X_train_v, y_train_tr, y_train_v = train_test_split(X_train_full,
                                                                y_train_full,
                                                                test_size=0.1)

```


```python
#prints the unique values in the y_train_full array, along with their counts.
print(np.unique(y_train_full, return_counts=True))
```

    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))



```python
#prints the shape of the X_train_tr
X_train_tr.shape
```




    (54000, 784)




```python
#prints the shape of the X_train_varray.
X_train_v.shape
```




    (6000, 784)




```python
# 2. Create model
model = keras.models.Sequential()
model.add(keras.layers.Dense(500, activation="relu", input_shape=(784,),
                             name="First_Hidden_Layer"))
model.add(keras.layers.Dense(500, activation="relu", name="Second_Hidden_Layer"))
model.add(keras.layers.Dense(10, activation="softmax", name="Output_Layer"))
```


```python
# This function prints a summary of the model, including the number of layers,
# the number of parameters in each layer, and the input and output shapes.
model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     First_Hidden_Layer (Dense)  (None, 500)               392500    
                                                                     
     Second_Hidden_Layer (Dense  (None, 500)               250500    
     )                                                               
                                                                     
     Output_Layer (Dense)        (None, 10)                5010      
                                                                     
    =================================================================
    Total params: 648010 (2.47 MB)
    Trainable params: 648010 (2.47 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________



```python
keras.utils.plot_model(model, show_shapes=True)
```




    
![png](/assets/img/MNIST_digits/output_18_0.png)
    




```python
model.layers[0].get_weights()
```




    [array([[-3.4882993e-02,  2.9719025e-03,  2.2934876e-02, ...,
              5.4532379e-02, -2.1777038e-02,  4.5120180e-02],
            [ 3.2115959e-02, -3.9733022e-02,  3.9428100e-02, ...,
             -7.0717484e-03, -6.1963797e-02,  3.8442224e-02],
            [ 3.5551213e-02,  2.2653654e-02,  2.4361566e-02, ...,
             -7.0976317e-03, -2.9596034e-02,  5.5925421e-02],
            ...,
            [-4.3508120e-02, -2.3868799e-02,  5.2154064e-08, ...,
              2.1156102e-02, -1.7477572e-04,  5.4189377e-02],
            [-1.6668070e-02, -2.9112604e-02, -4.9034886e-02, ...,
             -6.6048197e-02,  4.2988442e-02, -6.3812278e-02],
            [-2.4738345e-02,  4.2402938e-02,  9.7221732e-03, ...,
             -2.4722844e-02, -4.2742424e-02, -2.6337802e-05]], dtype=float32),
     array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.], dtype=float32)]




```python
#compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
```


```python
y_train_tr[0]
```




    2




```python
#training the model
history = model.fit(X_train_tr, y_train_tr, epochs=30, batch_size=32,
                    validation_data=(X_train_v, y_train_v))
```

    Epoch 1/30
    1688/1688 [==============================] - 13s 7ms/step - loss: 0.6031 - accuracy: 0.8517 - val_loss: 0.3316 - val_accuracy: 0.9038
    Epoch 2/30
    1688/1688 [==============================] - 12s 7ms/step - loss: 0.2821 - accuracy: 0.9205 - val_loss: 0.2572 - val_accuracy: 0.9272
    Epoch 3/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.2300 - accuracy: 0.9351 - val_loss: 0.2237 - val_accuracy: 0.9382
    Epoch 4/30
    1688/1688 [==============================] - 11s 6ms/step - loss: 0.1961 - accuracy: 0.9442 - val_loss: 0.1929 - val_accuracy: 0.9445
    Epoch 5/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.1708 - accuracy: 0.9513 - val_loss: 0.1742 - val_accuracy: 0.9498
    Epoch 6/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.1515 - accuracy: 0.9575 - val_loss: 0.1568 - val_accuracy: 0.9557
    Epoch 7/30
    1688/1688 [==============================] - 12s 7ms/step - loss: 0.1360 - accuracy: 0.9612 - val_loss: 0.1483 - val_accuracy: 0.9567
    Epoch 8/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.1229 - accuracy: 0.9643 - val_loss: 0.1474 - val_accuracy: 0.9582
    Epoch 9/30
    1688/1688 [==============================] - 11s 6ms/step - loss: 0.1122 - accuracy: 0.9677 - val_loss: 0.1284 - val_accuracy: 0.9652
    Epoch 10/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.1024 - accuracy: 0.9711 - val_loss: 0.1194 - val_accuracy: 0.9668
    Epoch 11/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0944 - accuracy: 0.9734 - val_loss: 0.1178 - val_accuracy: 0.9662
    Epoch 12/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0868 - accuracy: 0.9757 - val_loss: 0.1112 - val_accuracy: 0.9683
    Epoch 13/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0803 - accuracy: 0.9776 - val_loss: 0.1055 - val_accuracy: 0.9712
    Epoch 14/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0746 - accuracy: 0.9796 - val_loss: 0.1012 - val_accuracy: 0.9718
    Epoch 15/30
    1688/1688 [==============================] - 18s 11ms/step - loss: 0.0692 - accuracy: 0.9806 - val_loss: 0.0998 - val_accuracy: 0.9710
    Epoch 16/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0644 - accuracy: 0.9820 - val_loss: 0.0966 - val_accuracy: 0.9718
    Epoch 17/30
    1688/1688 [==============================] - 12s 7ms/step - loss: 0.0603 - accuracy: 0.9836 - val_loss: 0.0948 - val_accuracy: 0.9720
    Epoch 18/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0560 - accuracy: 0.9853 - val_loss: 0.0905 - val_accuracy: 0.9728
    Epoch 19/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0528 - accuracy: 0.9859 - val_loss: 0.0908 - val_accuracy: 0.9730
    Epoch 20/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0495 - accuracy: 0.9871 - val_loss: 0.0888 - val_accuracy: 0.9745
    Epoch 21/30
    1688/1688 [==============================] - 10s 6ms/step - loss: 0.0464 - accuracy: 0.9874 - val_loss: 0.0860 - val_accuracy: 0.9737
    Epoch 22/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0432 - accuracy: 0.9889 - val_loss: 0.0845 - val_accuracy: 0.9743
    Epoch 23/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0408 - accuracy: 0.9899 - val_loss: 0.0825 - val_accuracy: 0.9748
    Epoch 24/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0382 - accuracy: 0.9908 - val_loss: 0.0863 - val_accuracy: 0.9742
    Epoch 25/30
    1688/1688 [==============================] - 12s 7ms/step - loss: 0.0360 - accuracy: 0.9916 - val_loss: 0.0802 - val_accuracy: 0.9762
    Epoch 26/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0340 - accuracy: 0.9917 - val_loss: 0.0804 - val_accuracy: 0.9755
    Epoch 27/30
    1688/1688 [==============================] - 12s 7ms/step - loss: 0.0319 - accuracy: 0.9926 - val_loss: 0.0781 - val_accuracy: 0.9762
    Epoch 28/30
    1688/1688 [==============================] - 10s 6ms/step - loss: 0.0303 - accuracy: 0.9930 - val_loss: 0.0777 - val_accuracy: 0.9763
    Epoch 29/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0287 - accuracy: 0.9937 - val_loss: 0.0794 - val_accuracy: 0.9768
    Epoch 30/30
    1688/1688 [==============================] - 11s 7ms/step - loss: 0.0266 - accuracy: 0.9945 - val_loss: 0.0760 - val_accuracy: 0.9772



```python
history.params
```




    {'verbose': 1, 'epochs': 30, 'steps': 1688}




```python
history.epoch
```




    [0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24,
     25,
     26,
     27,
     28,
     29]




```python
history.history
```




    {'loss': [0.6031011939048767,
      0.28212636709213257,
      0.22998639941215515,
      0.19611385464668274,
      0.1708204746246338,
      0.1514693647623062,
      0.13597428798675537,
      0.12290101498365402,
      0.11216510832309723,
      0.10241560637950897,
      0.09438493102788925,
      0.08679485321044922,
      0.08029980212450027,
      0.07458988577127457,
      0.06923947483301163,
      0.06436340510845184,
      0.06025362014770508,
      0.05602743476629257,
      0.052779752761125565,
      0.049471765756607056,
      0.046374980360269547,
      0.043239858001470566,
      0.04076765850186348,
      0.038189150393009186,
      0.035983581095933914,
      0.03403782472014427,
      0.031875334680080414,
      0.030255533754825592,
      0.028664803132414818,
      0.02661091461777687],
     'accuracy': [0.8516851663589478,
      0.9204815030097961,
      0.9350555539131165,
      0.9441666603088379,
      0.9512962698936462,
      0.9574999809265137,
      0.9611851572990417,
      0.9643148183822632,
      0.9676666855812073,
      0.9711111187934875,
      0.9734073877334595,
      0.9757037162780762,
      0.9776111245155334,
      0.979629635810852,
      0.9805926084518433,
      0.9820370078086853,
      0.9836111068725586,
      0.9852592349052429,
      0.9859444499015808,
      0.9871296286582947,
      0.9874444603919983,
      0.9889259338378906,
      0.9898703694343567,
      0.9907962679862976,
      0.9915740489959717,
      0.9917407631874084,
      0.9926296472549438,
      0.9930185079574585,
      0.9936851859092712,
      0.9944629669189453],
     'val_loss': [0.3315768539905548,
      0.25722557306289673,
      0.2237492948770523,
      0.19288508594036102,
      0.17419828474521637,
      0.15677352249622345,
      0.1482946276664734,
      0.14735795557498932,
      0.12840957939624786,
      0.11935636401176453,
      0.11778061091899872,
      0.11123910546302795,
      0.10552014410495758,
      0.1012272760272026,
      0.09975381195545197,
      0.0966140553355217,
      0.09482162445783615,
      0.09049881994724274,
      0.09076787531375885,
      0.08875288814306259,
      0.08602162450551987,
      0.0845419242978096,
      0.08246560394763947,
      0.08632297068834305,
      0.08024030178785324,
      0.08037221431732178,
      0.0780949592590332,
      0.07772035896778107,
      0.07943751662969589,
      0.07598227262496948],
     'val_accuracy': [0.9038333296775818,
      0.9271666407585144,
      0.9381666779518127,
      0.9445000290870667,
      0.949833333492279,
      0.9556666612625122,
      0.9566666483879089,
      0.9581666588783264,
      0.9651666879653931,
      0.9668333530426025,
      0.9661666750907898,
      0.9683333039283752,
      0.9711666703224182,
      0.971833348274231,
      0.9710000157356262,
      0.971833348274231,
      0.972000002861023,
      0.9728333353996277,
      0.9729999899864197,
      0.9745000004768372,
      0.9736666679382324,
      0.9743333458900452,
      0.9748333096504211,
      0.9741666913032532,
      0.9761666655540466,
      0.9754999876022339,
      0.9761666655540466,
      0.9763333201408386,
      0.9768333435058594,
      0.9771666526794434]}




```python
history.history.keys()
```




    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])




```python
# plot graph
pd.DataFrame(history.history)
pd.DataFrame(history.history).plot(figsize=(10,7))
```




    <Axes: >




    
![png](/assets/img/MNIST_digits/output_27_1.png)
    





```python
histories = pd.DataFrame(history.history)
histories
```





  <div id="df-da243f13-95d6-4c91-89de-a301167327c6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>accuracy</th>
      <th>val_loss</th>
      <th>val_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.603101</td>
      <td>0.851685</td>
      <td>0.331577</td>
      <td>0.903833</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.282126</td>
      <td>0.920482</td>
      <td>0.257226</td>
      <td>0.927167</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.229986</td>
      <td>0.935056</td>
      <td>0.223749</td>
      <td>0.938167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.196114</td>
      <td>0.944167</td>
      <td>0.192885</td>
      <td>0.944500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.170820</td>
      <td>0.951296</td>
      <td>0.174198</td>
      <td>0.949833</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.151469</td>
      <td>0.957500</td>
      <td>0.156774</td>
      <td>0.955667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.135974</td>
      <td>0.961185</td>
      <td>0.148295</td>
      <td>0.956667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.122901</td>
      <td>0.964315</td>
      <td>0.147358</td>
      <td>0.958167</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.112165</td>
      <td>0.967667</td>
      <td>0.128410</td>
      <td>0.965167</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.102416</td>
      <td>0.971111</td>
      <td>0.119356</td>
      <td>0.966833</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.094385</td>
      <td>0.973407</td>
      <td>0.117781</td>
      <td>0.966167</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.086795</td>
      <td>0.975704</td>
      <td>0.111239</td>
      <td>0.968333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.080300</td>
      <td>0.977611</td>
      <td>0.105520</td>
      <td>0.971167</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.074590</td>
      <td>0.979630</td>
      <td>0.101227</td>
      <td>0.971833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.069239</td>
      <td>0.980593</td>
      <td>0.099754</td>
      <td>0.971000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.064363</td>
      <td>0.982037</td>
      <td>0.096614</td>
      <td>0.971833</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.060254</td>
      <td>0.983611</td>
      <td>0.094822</td>
      <td>0.972000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.056027</td>
      <td>0.985259</td>
      <td>0.090499</td>
      <td>0.972833</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.052780</td>
      <td>0.985944</td>
      <td>0.090768</td>
      <td>0.973000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.049472</td>
      <td>0.987130</td>
      <td>0.088753</td>
      <td>0.974500</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.046375</td>
      <td>0.987444</td>
      <td>0.086022</td>
      <td>0.973667</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.043240</td>
      <td>0.988926</td>
      <td>0.084542</td>
      <td>0.974333</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.040768</td>
      <td>0.989870</td>
      <td>0.082466</td>
      <td>0.974833</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.038189</td>
      <td>0.990796</td>
      <td>0.086323</td>
      <td>0.974167</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.035984</td>
      <td>0.991574</td>
      <td>0.080240</td>
      <td>0.976167</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.034038</td>
      <td>0.991741</td>
      <td>0.080372</td>
      <td>0.975500</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.031875</td>
      <td>0.992630</td>
      <td>0.078095</td>
      <td>0.976167</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.030256</td>
      <td>0.993019</td>
      <td>0.077720</td>
      <td>0.976333</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.028665</td>
      <td>0.993685</td>
      <td>0.079438</td>
      <td>0.976833</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.026611</td>
      <td>0.994463</td>
      <td>0.075982</td>
      <td>0.977167</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-da243f13-95d6-4c91-89de-a301167327c6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-da243f13-95d6-4c91-89de-a301167327c6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-da243f13-95d6-4c91-89de-a301167327c6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a835f81d-6e67-4686-abf4-0670b763c4b6">
  <button class="colab-df-quickchart" onclick="quickchart('df-a835f81d-6e67-4686-abf4-0670b763c4b6')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a835f81d-6e67-4686-abf4-0670b763c4b6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
#Checking test accuracy
model.evaluate(X_test, y_test)
```

    313/313 [==============================] - 1s 5ms/step - loss: 0.0695 - accuracy: 0.9779





    [0.06946521252393723, 0.9779000282287598]




```python
# How to predict other cases
import matplotlib.pyplot as plt
plt.imshow(X_train_full[8].reshape(28,28), cmap='binary')
```




    <matplotlib.image.AxesImage at 0x780c35ebdf60>




    
![png](/assets/img/MNIST_digits/output_31_1.png)
    



```python
model.predict(X_train_full[:1])
```

    1/1 [==============================] - 0s 65ms/step





    array([[2.9947158e-09, 2.3034241e-08, 7.6127554e-07, 2.9612569e-02,
            1.8399178e-15, 9.7038627e-01, 2.9789549e-11, 7.4470265e-08,
            7.4396831e-09, 1.9115635e-07]], dtype=float32)




```python
import matplotlib.pyplot as plt

# Get the predictions from the model
predictions = np.argmax(model.predict(X_test), axis=-1)
# Find the indexes of the misclassified images
misclassifiedIndexes = []
index = 0
for target, predict in zip(y_test, predictions):
  if target != predict:
    misclassifiedIndexes.append(index)
  index += 1

misclassifiedIndexes

plt.figure(figsize=(25,4))
for index in range(5):
  plt.subplot(1,5,index+1)
  plt.imshow(X_test[misclassifiedIndexes[index]].reshape(28,28), cmap='binary')
  plt.axis('off')
  plt.title(f'Target: {y_test[misclassifiedIndexes[index]]}    Predicted: {predictions[misclassifiedIndexes[index]]}', fontsize=14)

plt.show()
```

    313/313 [==============================] - 1s 3ms/step



    
![png](/assets/img/MNIST_digits/output_33_1.png)
    



```python
# Import necessary libraries
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=-1))

# Define your classes (replace with your actual class labels)
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Create a DataFrame for the confusion matrix
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# Create a heatmap with annotations
plt.figure(figsize=(10, 10))
sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.ylabel('True class', color='white')
plt.xlabel('Predicted class', color='white')
plt.tick_params(color='white', labelcolor='white')
plt.show()

```

    313/313 [==============================] - 1s 3ms/step



    
![png](/assets/img/MNIST_digits/output_34_1.png)
    



```python
# Assuming you have defined 'model', 'X_test', 'y_test', and 'classes' elsewhere in your code

# Generate predictions for the test data using your model
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame from the confusion matrix
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')  # Customize the heatmap appearance
plt.ylabel('True class')
plt.xlabel('Predicted class')

# Customize the tick marks and labels color to black
plt.tick_params(color='black', labelcolor='black')

# Show the plot
plt.show()

```

    313/313 [==============================] - 1s 3ms/step



    
![png](/assets/img/MNIST_digits/output_35_1.png)
    


#Part 1
The model did not overfit. The training loss of 0.026611 and training accuracy of 99.45%, it shows that the model has learned the training data very well. The validation loss of 0.075982 and validation accuracy of 97.72% which indaicates that the model performed well on the validarion data.

The model was then put to the test on a sepeerat testing dataset, it achieved a test accuracy of 97.79%, which is consistent with the validation accuracy. This consistency suggests that the model hasn't overfit the training data, as it performs similarly on both validation and test datasets. The test loss of 0.0695.


```python

#additional layer of 512 neurons
#increased the number of neurons in the other layers
# Create sequesntial model
model2 = keras.models.Sequential()

model2.add(keras.layers.Dense(512, activation="relu", input_shape=(784,),
                             name="First_Hidden_Layer"))

model2.add(keras.layers.Dense(512, activation="relu",
                             name="Second_Hidden_Layer"))

model2.add(keras.layers.Dense(512, activation='relu',
                 name='Third_Hidden_Layer'))

model2.add(keras.layers.Dense(10, activation="softmax",
                             name="Output_Layer"))
```


```python
model2.summary()
```

    Model: "sequential_8"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     First_Hidden_Layer (Dense)  (None, 512)               401920    
                                                                     
     Second_Hidden_Layer (Dense  (None, 512)               262656    
     )                                                               
                                                                     
     Third_Hidden_Layer (Dense)  (None, 512)               262656    
                                                                     
     Output_Layer (Dense)        (None, 10)                5130      
                                                                     
    =================================================================
    Total params: 932362 (3.56 MB)
    Trainable params: 932362 (3.56 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________



```python
# model plot
keras.utils.plot_model(model2, show_shapes=True)
```




    
![png](/assets/img/MNIST_digits/output_39_0.png)
    




```python
# Compile model
model2.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
```


```python
#training the model
history = model2.fit(X_train_tr, y_train_tr, epochs=30, batch_size=32,
                    validation_data=(X_train_v, y_train_v))
```

    Epoch 1/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.5938 - accuracy: 0.8497 - val_loss: 0.3004 - val_accuracy: 0.9147
    Epoch 2/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.2535 - accuracy: 0.9269 - val_loss: 0.2262 - val_accuracy: 0.9347
    Epoch 3/30
    1688/1688 [==============================] - 17s 10ms/step - loss: 0.1972 - accuracy: 0.9428 - val_loss: 0.1891 - val_accuracy: 0.9472
    Epoch 4/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.1626 - accuracy: 0.9532 - val_loss: 0.1668 - val_accuracy: 0.9515
    Epoch 5/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.1379 - accuracy: 0.9611 - val_loss: 0.1506 - val_accuracy: 0.9582
    Epoch 6/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.1192 - accuracy: 0.9654 - val_loss: 0.1323 - val_accuracy: 0.9623
    Epoch 7/30
    1688/1688 [==============================] - 17s 10ms/step - loss: 0.1035 - accuracy: 0.9702 - val_loss: 0.1182 - val_accuracy: 0.9653
    Epoch 8/30
    1688/1688 [==============================] - 16s 10ms/step - loss: 0.0906 - accuracy: 0.9734 - val_loss: 0.1127 - val_accuracy: 0.9672
    Epoch 9/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0806 - accuracy: 0.9769 - val_loss: 0.1072 - val_accuracy: 0.9685
    Epoch 10/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0716 - accuracy: 0.9794 - val_loss: 0.0999 - val_accuracy: 0.9700
    Epoch 11/30
    1688/1688 [==============================] - 17s 10ms/step - loss: 0.0634 - accuracy: 0.9817 - val_loss: 0.0975 - val_accuracy: 0.9690
    Epoch 12/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0571 - accuracy: 0.9843 - val_loss: 0.0927 - val_accuracy: 0.9732
    Epoch 13/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0508 - accuracy: 0.9857 - val_loss: 0.0896 - val_accuracy: 0.9737
    Epoch 14/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0455 - accuracy: 0.9878 - val_loss: 0.0846 - val_accuracy: 0.9752
    Epoch 15/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0409 - accuracy: 0.9887 - val_loss: 0.0864 - val_accuracy: 0.9732
    Epoch 16/30
    1688/1688 [==============================] - 16s 10ms/step - loss: 0.0368 - accuracy: 0.9902 - val_loss: 0.0849 - val_accuracy: 0.9735
    Epoch 17/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0329 - accuracy: 0.9915 - val_loss: 0.0812 - val_accuracy: 0.9757
    Epoch 18/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0294 - accuracy: 0.9927 - val_loss: 0.0851 - val_accuracy: 0.9752
    Epoch 19/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0261 - accuracy: 0.9935 - val_loss: 0.0779 - val_accuracy: 0.9773
    Epoch 20/30
    1688/1688 [==============================] - 16s 10ms/step - loss: 0.0231 - accuracy: 0.9951 - val_loss: 0.0823 - val_accuracy: 0.9760
    Epoch 21/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0210 - accuracy: 0.9956 - val_loss: 0.0778 - val_accuracy: 0.9765
    Epoch 22/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0190 - accuracy: 0.9960 - val_loss: 0.0804 - val_accuracy: 0.9767
    Epoch 23/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0171 - accuracy: 0.9970 - val_loss: 0.0779 - val_accuracy: 0.9773
    Epoch 24/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0152 - accuracy: 0.9975 - val_loss: 0.0811 - val_accuracy: 0.9757
    Epoch 25/30
    1688/1688 [==============================] - 17s 10ms/step - loss: 0.0137 - accuracy: 0.9979 - val_loss: 0.0782 - val_accuracy: 0.9768
    Epoch 26/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0124 - accuracy: 0.9981 - val_loss: 0.0779 - val_accuracy: 0.9783
    Epoch 27/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0110 - accuracy: 0.9986 - val_loss: 0.0771 - val_accuracy: 0.9777
    Epoch 28/30
    1688/1688 [==============================] - 16s 9ms/step - loss: 0.0100 - accuracy: 0.9988 - val_loss: 0.0783 - val_accuracy: 0.9782
    Epoch 29/30
    1688/1688 [==============================] - 17s 10ms/step - loss: 0.0091 - accuracy: 0.9991 - val_loss: 0.0770 - val_accuracy: 0.9783
    Epoch 30/30
    1688/1688 [==============================] - 15s 9ms/step - loss: 0.0083 - accuracy: 0.9991 - val_loss: 0.0795 - val_accuracy: 0.9785



```python

```


```python
#Checking test accuracy
model2.evaluate(X_test, y_test)
```

    313/313 [==============================] - 1s 3ms/step - loss: 0.0710 - accuracy: 0.9782





    [0.07104338705539703, 0.9782000184059143]



#Part3
 Model2 slightly performed better than model one. Model to has a training loss of 0.0083 and a training accuracy of 99.91% show that the model learned the training data very well.

 The validation results show a validation loss of 0.0795 and validation accuracy of 97.85%, indicating that the model generalizes effectively to new data

Model 2 maintains a high level of performance with a test accuracy of 97.82%. The test loss of 0.0710 is in line with the overall strong performance of the model.
