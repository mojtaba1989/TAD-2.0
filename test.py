import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
from keras.applications.mobilenet import MobileNet


def myInception(layer_in, f1, f2, f3, f4, identity=False):
    conv1_1 = layers.Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(layer_in)
    conv1_2 = layers.Conv2D(filters=f2, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    conv1_3 = layers.Conv2D(filters=f3, kernel_size=(3, 3), padding='same', activation='relu')(conv1_2)
    pool = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = layers.Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(pool)
    layer_out = layers.concatenate([conv1_1, conv1_2, conv1_3, pool])
    if identity:
        merge_in = layer_in
        if layer_in.shape[-1] != f1 + f2 + f3 + f4:
            merge_input = layers.Conv2D(f1 + f2 + f3 + f4, (1, 1), padding='same', activation='relu')(layer_in)
        layer_out = layers.add([layer_out, merge_input])
        layer_out = layers.Activation('relu')(layer_out)
    return layer_out

run_meta = tf.RunMetadata()
# with tf.Session(graph=tf.Graph()) as sess:
#     K.set_session(sess)
#     with tf.device('/cpu:0'):
#         # base_model = MobileNet(alpha=1, weights=None, input_tensor=tf.placeholder('float32', shape=(1,224,224,3)))
#         input = layers.Input(shape=(75, 171, 3), tensor=tf.placeholder('float32', shape=(1, 75, 171, 3)))
#         x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation="relu")(input)
#         x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
#         x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
#         x = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
#         x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
#         x = myInception(x, 64, 128, 32, 32)
#         x = myInception(x, 128, 192, 96, 64)
#         x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
#         x = myInception(x, 192, 208, 48, 64)
#         x = myInception(x, 160, 224, 64, 64)
#         x = myInception(x, 128, 256, 64, 64)
#         x = myInception(x, 112, 288, 64, 64)
#         x = myInception(x, 256, 320, 128, 128)
#         x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
#         x = layers.Flatten()(x)
#         x = layers.Dense(200)(x)
#         x = layers.Activation(activation="relu")(x)
#         x = layers.BatchNormalization(axis=-1)(x)
#         x = layers.Dropout(0.5)(x)
#         x = layers.Dense(20, activation="relu")(x)
#         x = layers.Dense(10, activation="relu")(x)
#         x = layers.Dense(1, activation="linear")(x)
#         base_model = keras.models.Model(inputs=input, outputs=x)
#         opts = tf.profiler.ProfileOptionBuilder.float_operation()
#         flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
#         opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
#         params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
# print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

# with tf.Session(graph=tf.Graph()) as sess:
#     K.set_session(sess)
#     with tf.device('/cpu:0'):
#         # base_model = MobileNet(alpha=1, weights=None, input_tensor=tf.placeholder('float32', shape=(1,224,224,3)))
#         from tensorflow.keras.applications.inception_v3 import InceptionV3
#
#         temp = InceptionV3(input_shape=(75, 171, 3),
#                            include_top=False,
#                            weights=None,
#                            input_tensor=tf.placeholder('float32', shape=(1,75, 171, 3)))
#         x = layers.DepthwiseConv2D(kernel_size=(1, 4))(temp.output)
#         x = layers.Flatten()(x)
#         x = layers.Dense(200)(x)
#         x = layers.Activation(activation="relu")(x)
#         x = layers.BatchNormalization(axis=-1)(x)
#         x = layers.Dense(20, activation="relu")(x)
#         x = layers.Dense(10, activation="relu")(x)
#         x = layers.Dense(1, activation="linear")(x)
#         model = keras.models.Model(inputs=temp.input, outputs=x)
#         opts = tf.profiler.ProfileOptionBuilder.float_operation()
#         flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
#         opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
#         params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
# print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    with tf.device('/cpu:0'):
        # base_model = MobileNet(alpha=1, weights=None, input_tensor=tf.placeholder('float32', shape=(1,224,224,3)))

        from tensorflow.keras.applications.vgg16 import VGG16 as PTModel

        temp = PTModel(input_shape=(75, 171, 3),
                           include_top=False,
                           weights=None,
                           input_tensor=tf.keras.layers.Input(shape=(75, 171, 3),
                                                              tensor=tf.ones(shape=(1, 75, 171, 3))))
        x = layers.GlobalAveragePooling2D()(temp.output)
        x = layers.Dense(200)(x)
        x = layers.Activation(activation="relu")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dense(10, activation="relu")(x)
        x = layers.Dense(1, activation="linear")(x)
        model = keras.models.Model(inputs=temp.input, outputs=x)
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))