# -*- coding: utf-8 -*-

def build_model(input_shape, num_chars, num_classes):
    print('🧠 Modelo criado')
    import tensorflow as tf
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = [tf.keras.layers.Dense(num_classes, activation='softmax', name=f'dense_{i+1}')(x) for i in range(num_chars)]
    return tf.keras.Model(inputs=inputs, outputs=outputs)