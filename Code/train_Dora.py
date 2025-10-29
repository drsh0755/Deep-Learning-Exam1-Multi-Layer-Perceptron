
# Add this at the VERY BEGINNING of your script, before any other imports
# This must be done BEFORE importing TensorFlow

import os
import sys

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only

# Suppress CUDA and cuDNN warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Suppress XLA warnings (if using XLA compilation)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Now import TensorFlow and other libraries
import tensorflow as tf
import logging

# Set TensorFlow logging level
tf.get_logger().setLevel(logging.ERROR)

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Optional: Suppress absl logging
logging.getLogger('absl').setLevel(logging.ERROR)

# ============================================================
# Now your actual imports
# ============================================================
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
logging.getLogger('absl').setLevel(logging.ERROR)


#------------------------------------------------------------------------------------------------------------------

'''
LAST UPDATE 10/20/2021 LSDR
last update 10/21/2021 lsdr
02/14/2022 am LSDR CHECK CONSISTENCY
02/14/2022 pm LSDR Change result for results

'''
#------------------------------------------------------------------------------------------------------------------

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("../..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 50
BATCH_SIZE = 128

## Image processing
CHANNELS = 3
IMAGE_SIZE = 128

NICKNAME = 'Dora'
#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return:
    '''


    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = xdf_data['target'].apply(x)

        final_target = to_categorical(list(final_target))

        xfinal=[]
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in  (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        xdf_data['target_class'] = final_target


    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))

        xdepth = len(class_names)

        final_target = tf.one_hot(target, xdepth)

        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            final_target = xfinal

        xdf_data['target_class'] = final_target

    if target_type == 3:
        # target_class is already done
        pass

    return class_names


# ------------------------------------------------------------------------------------------------------------------
# ✅ NEW FUNCTION: Check and Report Class Imbalance
# ------------------------------------------------------------------------------------------------------------------

def check_class_imbalance(train_data, class_names):
    '''
    Check for class imbalance in training data
    Returns class_weight dictionary if imbalance detected
    '''

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Get class counts
    class_counts = train_data['target'].value_counts().sort_index()

    print("\nClass Distribution (Training Data):")
    print("-" * 70)
    for class_name in sorted(class_counts.index):
        count = class_counts[class_name]
        pct = (count / len(train_data) * 100)
        print(f"  {class_name}: {count:6d} samples ({pct:6.2f}%)")

    # Calculate imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count

    print("-" * 70)
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}x (max/min)")

    # Categorize imbalance severity
    if imbalance_ratio < 1.5:
        severity = "✓ BALANCED"
    elif imbalance_ratio < 3:
        severity = "⚠ MILD IMBALANCE"
    elif imbalance_ratio < 10:
        severity = "⚠ MODERATE IMBALANCE"
    else:
        severity = "✗ SEVERE IMBALANCE"

    print(f"Status: {severity}")
    print("=" * 70)

    # Calculate class weights if imbalanced
    if imbalance_ratio >= 1.5:
        y_train = train_data['target'].values

        class_weights_array = compute_class_weight(
            'balanced',
            classes=class_names,
            y=y_train
        )

        class_weight_dict = {}
        for idx, class_name in enumerate(class_names):
            class_weight_dict[idx] = class_weights_array[idx]

        print("\nClass Weights (for balancing):")
        print("-" * 70)
        for idx, class_name in enumerate(class_names):
            weight = class_weight_dict[idx]
            print(f"  Class {idx} ({class_name}): {weight:.4f}")
        print("=" * 70)

        return class_weight_dict
    else:
        print("\nNo significant imbalance detected. Proceeding without class weights.")
        print("=" * 70)
        return None

#------------------------------------------------------------------------------------------------------------------

def process_path(feature, target, augment=False):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    label = target

    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])

    # augmentation
    # NORMALIZATION (always, not just augmented)
    img = img / 255.0
    # DATA AUGMENTATION (only for training, not testing)
    if augment:
        # Random rotation
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

        # Random horizontal flip
        img = tf.image.random_flip_left_right(img)

        img = tf.image.random_flip_up_down(img)

        # Random brightness adjustment
        img = tf.image.random_brightness(img, max_delta=0.2)

        # Random contrast adjustment
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)


    img = tf.reshape(img, [-1])

    return img, label
#------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

    return y_target
#------------------------------------------------------------------------------------------------------------------


def read_data(num_classes, augment=False):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(lambda x, y: process_path(x, y, augment=augment),num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds
#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
#------------------------------------------------------------------------------------------------------------------

def model_definition():
    # Define a Keras sequential model
    model = tf.keras.Sequential()

    # Define the first dense layer
    # ✅ ADD L2 regularization + Dropout to each layer
    model.add(tf.keras.layers.Dense(1024, activation='relu', input_shape=(INPUTS_r,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(100, activation='relu'))
    # # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(80, activation='relu'))
    # # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(50, activation='relu'))
    # # # model.add(tf.keras.layers.Dropout(0.05))

    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax')) #final layer , outputs_a is the number of targets

    model.compile(optimizer='AdamW', loss='categorical_crossentropy', metrics=['accuracy'])

    save_model(model) #print Summary
    return model
#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds):
    # '''
    #     train the model
    # '''
    #
    # #early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience =100)
    # check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.keras'.format(NICKNAME), monitor='accuracy', save_best_only=True)
    # final_model = model_definition()
    #
    # #final_model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point])
    # final_model.fit(train_ds,  epochs=n_epoch, callbacks=[check_point])
    '''
            train the model

            Args:
                train_ds: training dataset
                class_weights: optional dictionary of class weights for imbalanced data
    '''

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience =100)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.keras'.format(NICKNAME), monitor='accuracy',
                                                     save_best_only=True)
    final_model = model_definition()

    # ✅ MODIFIED: Pass class_weights if provided
    fit_kwargs = {
        'epochs': n_epoch,
        'callbacks': [check_point]
    }

    # if class_weights is not None:
    #     print("\nApplying class weights to balance training...")
    #     fit_kwargs['class_weight'] = class_weights
    #
    # final_model.fit(train_ds, **fit_kwargs)
    final_model.fit(train_ds, epochs=n_epoch, callbacks=[check_point])


#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME))
    res = final_model.predict(test_ds)
    xres = [ tf.argmax(f).numpy() for f in res]
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res


    # For multiclass

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # End of Multiclass

    xcont = 1
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        else:
            xmet =print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)
    # Ask for arguments for each metric
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    train_ds = read_data( OUTPUTS_a, augment=True)
    train_func(train_ds)
    # # ✅ NEW: Check for class imbalance and get class weights if needed
    # class_weights = check_class_imbalance(xdf_dset, class_names)
    #
    # train_ds = read_data(OUTPUTS_a, augment=True)
    # # ✅ MODIFIED: Pass class_weights to training function
    # train_func(train_ds, class_weights=class_weights)

    # Preprocessing Test dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()

    test_ds= read_data(OUTPUTS_a, augment=False)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'coh']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------

