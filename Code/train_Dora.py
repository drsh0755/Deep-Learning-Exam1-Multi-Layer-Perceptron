# ============================================================
# TRAIN_DORA.PY - COMPLETE WITH ADVANCED ARCHITECTURES
# All architectures built-in, no separate file needed!
# ============================================================

# Suppress TensorFlow logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

logging.getLogger('absl').setLevel(logging.ERROR)

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model, regularizers  # <-- ADDED

# ============================================================
# CONFIGURATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

# Image processing
CHANNELS = 3
IMAGE_SIZE = 128  # Changed from 200

# Dataset paths
OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

n_epoch = 1000  # Increased from 50
BATCH_SIZE = 64

NICKNAME = 'Dora'


# ============================================================
# ADVANCED ARCHITECTURE DEFINITIONS (REPLACED)
# ============================================================

def model_residual_mlp(INPUTS_r, OUTPUTS_a):
    '''Residual MLP with skip connections'''

    input_layer = layers.Input(shape=(INPUTS_r,))
    reg = regularizers.l2(0.00005)

    # Block 1
    x = layers.Dense(512, activation=None, kernel_regularizer=reg)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(512, activation=None, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    residual = layers.Dense(512, kernel_regularizer=reg)(input_layer)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    residual = x
    x = layers.Dense(256, activation=None, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation=None, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    residual = layers.Dense(256, kernel_regularizer=reg)(residual)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    residual = x
    x = layers.Dense(128, activation=None, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation=None, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    residual = layers.Dense(128, kernel_regularizer=reg)(residual)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)

    output = layers.Dense(OUTPUTS_a, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output, name='ResidualMLP')

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0005,
        weight_decay=0.0001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def model_dense_mlp(INPUTS_r, OUTPUTS_a):
    '''Dense MLP with concatenation connections'''

    input_layer = layers.Input(shape=(INPUTS_r,))
    reg = regularizers.l2(0.00005)

    # Dense Block 1
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    features = [input_layer, x]

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(layers.Concatenate()(features))
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    features.append(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(layers.Concatenate()(features))
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    features.append(x)

    # Dense Block 2
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(layers.Concatenate()(features))
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    features = [x]

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    features.append(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(layers.Concatenate()(features))
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    features.append(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(layers.Concatenate()(features))
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    output = layers.Dense(OUTPUTS_a, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output, name='DenseMLP')

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0005,
        weight_decay=0.0001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def model_mixture_of_experts(INPUTS_r, OUTPUTS_a):
    '''Mixture of Experts MLP'''

    input_layer = layers.Input(shape=(INPUTS_r,))
    reg = regularizers.l2(0.00005)
    num_experts = 4

    # Experts
    expert1 = layers.Dense(256, activation='relu', kernel_regularizer=reg)(input_layer)
    expert1 = layers.BatchNormalization()(expert1)
    expert1 = layers.Dropout(0.3)(expert1)
    expert1 = layers.Dense(128, activation='relu', kernel_regularizer=reg)(expert1)
    expert1 = layers.BatchNormalization()(expert1)

    expert2 = layers.Dense(256, activation='relu', kernel_regularizer=reg)(input_layer)
    expert2 = layers.BatchNormalization()(expert2)
    expert2 = layers.Dropout(0.3)(expert2)
    expert2 = layers.Dense(128, activation='relu', kernel_regularizer=reg)(expert2)
    expert2 = layers.BatchNormalization()(expert2)

    expert3 = layers.Dense(256, activation='elu', kernel_regularizer=reg)(input_layer)
    expert3 = layers.BatchNormalization()(expert3)
    expert3 = layers.Dropout(0.3)(expert3)
    expert3 = layers.Dense(128, activation='elu', kernel_regularizer=reg)(expert3)
    expert3 = layers.BatchNormalization()(expert3)

    expert4 = layers.Dense(128, activation='relu', kernel_regularizer=reg)(input_layer)
    expert4 = layers.BatchNormalization()(expert4)
    expert4 = layers.Dropout(0.2)(expert4)

    # Gating Network
    gate = layers.Dense(256, activation='relu', kernel_regularizer=reg)(input_layer)
    gate = layers.BatchNormalization()(gate)
    gate = layers.Dense(num_experts, activation='softmax')(gate)

    # Combine
    # expert_outputs = layers.Stack(axis=1)([expert1, expert2, expert3, expert4])
    expert_outputs = layers.Lambda(lambda x: tf.stack(x, axis=1))([expert1, expert2, expert3, expert4]) #only for achitecture == mixture
    gate_expanded = layers.Reshape((num_experts, 1))(gate)
    gated = layers.Multiply()([expert_outputs, gate_expanded])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(gated)

    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(OUTPUTS_a, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output, name='MixtureOfExpertsMLP')

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0005,
        weight_decay=0.0001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def model_auxiliary_classifier(INPUTS_r, OUTPUTS_a):
    '''MLP with Auxiliary Classifiers'''

    input_layer = layers.Input(shape=(INPUTS_r,))
    reg = regularizers.l2(0.00005)

    x = layers.Dense(512, activation='relu', kernel_regularizer=reg)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Auxiliary 1
    aux1 = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    aux1 = layers.BatchNormalization()(aux1)
    aux1 = layers.Dense(OUTPUTS_a, activation='softmax', name='aux_output_1')(aux1)

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Auxiliary 2
    aux2 = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    aux2 = layers.BatchNormalization()(aux2)
    aux2 = layers.Dense(OUTPUTS_a, activation='softmax', name='aux_output_2')(aux2)

    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    output = layers.Dense(OUTPUTS_a, activation='softmax', name='main_output')(x)

    model = Model(
        inputs=input_layer,
        outputs=[output, aux1, aux2],
        name='AuxiliaryClassifierMLP'
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0005,
        weight_decay=0.0001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss={
            'main_output': 'categorical_crossentropy',
            'aux_output_1': 'categorical_crossentropy',
            'aux_output_2': 'categorical_crossentropy'
        },
        loss_weights={
            'main_output': 1.0,
            'aux_output_1': 0.3,
            'aux_output_2': 0.3
        },
        metrics=['accuracy']
    )

    return model


def model_hybrid_advanced(INPUTS_r, OUTPUTS_a):
    '''Ultimate Hybrid Architecture - RECOMMENDED'''

    input_layer = layers.Input(shape=(INPUTS_r,))
    reg = regularizers.l2(0.00005)
    original_input = input_layer

    # Block 1: Input Injection + Residual
    x = layers.Concatenate()([input_layer, input_layer])
    x = layers.Dense(512, activation=None, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(512, activation=None, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    residual = layers.Dense(512, kernel_regularizer=reg)(
        layers.Concatenate()([input_layer, input_layer])
    )
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Block 2: Dense + Input Injection
    original_injected = layers.Dense(512, kernel_regularizer=reg)(original_input)
    x = layers.Concatenate()([x, original_injected])
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    residual = x
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    residual_proj = layers.Dense(256, kernel_regularizer=reg)(residual)
    x = layers.Add()([x, residual_proj])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Block 3: Dense Concatenation + Input Injection
    original_injected = layers.Dense(256, kernel_regularizer=reg)(original_input)
    x = layers.Concatenate()([x, original_injected])
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    residual = x
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    residual_proj = layers.Dense(128, kernel_regularizer=reg)(residual)
    x = layers.Add()([x, residual_proj])
    x = layers.Activation('relu')(x)

    output = layers.Dense(OUTPUTS_a, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output, name='HybridAdvancedMLP')

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0005,
        weight_decay=0.0001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model(architecture_name, INPUTS_r, OUTPUTS_a):
    '''Select which architecture to use'''

    if architecture_name == 'residual':
        return model_residual_mlp(INPUTS_r, OUTPUTS_a)
    elif architecture_name == 'dense':
        return model_dense_mlp(INPUTS_r, OUTPUTS_a)
    elif architecture_name == 'mixture':
        return model_mixture_of_experts(INPUTS_r, OUTPUTS_a)
    elif architecture_name == 'auxiliary':
        return model_auxiliary_classifier(INPUTS_r, OUTPUTS_a)
    elif architecture_name == 'hybrid':
        return model_hybrid_advanced(INPUTS_r, OUTPUTS_a)
    else:
        print(f"Unknown architecture: {architecture_name}, using hybrid")
        return model_hybrid_advanced(INPUTS_r, OUTPUTS_a)


# ============================================================
# END OF ADVANCED ARCHITECTURES
# ============================================================

def process_target(target_type):
    '''Process target variable'''

    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:
        x = lambda x: tf.argmax(x == class_names).numpy()
        final_target = xdf_data['target'].apply(x)
        final_target = to_categorical(list(final_target))

        xfinal = []
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        xdf_data['target_class'] = final_target

    # (Other target_type logic from your script is here)

    return class_names


def check_class_imbalance(train_data, class_names):
    '''Check for class imbalance in training data'''

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)

    class_counts = train_data['target'].value_counts().sort_index()

    print("\nClass Distribution (Training Data):")
    print("-" * 70)
    for class_name in sorted(class_counts.index):
        count = class_counts[class_name]
        pct = (count / len(train_data) * 100)
        print(f"  {class_name}: {count:6d} samples ({pct:6.2f}%)")

    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count

    print("-" * 70)
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}x (max/min)")

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


def process_path(feature, target, augment=False):
    '''Process image path and return normalized image'''

    label = target
    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])

    img = img / 255.0

    if augment:
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)  # <-- ADDED
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)  # <-- ADDED

    img = tf.reshape(img, [-1])

    return img, label


def get_target(num_classes, xdf_dset):  # <-- UPDATED
    '''Get the target from the dataset'''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

    return y_target


def read_data(num_classes, xdf_dset, augment=False):  # <-- UPDATED
    '''Read dataset and process'''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes, xdf_dset)  # <-- UPDATED

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets))

    final_ds = list_ds.map(lambda x, y: process_path(x, y, augment=augment), num_parallel_calls=AUTOTUNE).batch(
        BATCH_SIZE)

    return final_ds


def save_model(model):
    '''Save model summary to file'''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def model_definition(INPUTS_r, OUTPUTS_a):
    '''Define and return model'''

    # ⭐ CHANGE THIS LINE TO TEST DIFFERENT ARCHITECTURES:
    # Options: 'residual', 'dense', 'mixture', 'auxiliary', 'hybrid'
    architecture = 'hybrid'

    print("\n" + "=" * 70)
    print(f"Loading {architecture.upper()} Architecture")
    print("=" * 70 + "\n")

    model = get_model(architecture, INPUTS_r, OUTPUTS_a)
    save_model(model)

    return model


def train_func(train_ds, class_weights=None):  # <-- REPLACED
    '''Train the model'''

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=25,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    check_point = tf.keras.callbacks.ModelCheckpoint(
        'model_{}.keras'.format(NICKNAME),
        monitor='accuracy',
        save_best_only=True,
        verbose=1
    )

    final_model = model_definition(INPUTS_r, OUTPUTS_a)

    fit_kwargs = {
        'epochs': n_epoch,
        'callbacks': [early_stop, lr_scheduler, check_point],
        'verbose': 1
    }

    if class_weights is not None:
        print("\n" + "=" * 70)
        print("Applying class weights to balance training...")
        print("=" * 70 + "\n")
        fit_kwargs['class_weight'] = class_weights

    history = final_model.fit(train_ds, **fit_kwargs)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Epochs trained: {len(history.history['accuracy'])}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Best accuracy: {max(history.history['accuracy']):.4f}")
    print("=" * 70 + "\n")


def predict_func(test_ds):  # <-- REPLACED
    '''Make predictions on test data'''

    final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME))

    predictions = final_model.predict(test_ds)

    # Handle both single and multiple outputs
    if isinstance(predictions, list):
        main_output = predictions[0]  # Get main output if using auxiliary model
        xres = [tf.argmax(f).numpy() for f in main_output]
    else:
        xres = [tf.argmax(f).numpy() for f in predictions]

    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)


def metrics_func(metrics, aggregates=[]):
    '''Calculate multiple evaluation metrics'''

    def f1_score_metric(y_true, y_pred, type):
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

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    xcont = 1
    xsum = 0

    for xm in metrics:
        if xm == 'f1_micro':
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            xmet = matthews_metric(y_true, y_pred)
        else:
            xmet = print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)


def main():  # <-- REPLACED
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset

    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)

    class_names = process_target(1)

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset
    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    # ✓ ENABLE CLASS WEIGHTS
    class_weights = check_class_imbalance(xdf_dset, class_names)

    train_ds = read_data(OUTPUTS_a, xdf_dset, augment=True)

    # ✓ PASS CLASS WEIGHTS
    train_func(train_ds, class_weights=class_weights)

    # Preprocessing Test dataset
    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()

    test_ds = read_data(OUTPUTS_a, xdf_dset, augment=False)
    predict_func(test_ds)

    ## Metrics Function
    list_of_metrics = ['f1_macro', 'coh', 'acc', 'f1_weighted', 'mat']
    list_of_agg = ['avg', 'sum']
    metrics_func(list_of_metrics, list_of_agg)


if __name__ == "__main__":
    main()