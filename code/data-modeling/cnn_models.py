# -*- coding: utf-8 -*-
import tensorflow as tf

tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)

KERNEL_INITIALIZER = "he_uniform"

mse_fn = tf.keras.losses.MeanSquaredError()


def mse_mvar(y_true, y_pred):
    e = y_true - y_pred[:, 0:1]
    e = tf.stop_gradient(e * e)
    return mse_fn(y_true, y_pred[:, 0:1]) + mse_fn(e, y_pred[:, 1:2])

def mse(y_true, y_pred):
    return mse_fn(y_true, y_pred[:, 0:1])

def mse_var(y_true, y_pred):
    e = y_true - y_pred[:, 0:1]
    return mse_fn(e, y_pred[:, 1:2])



def classic_convolutional_model(
    wide1,
    depth1,
    wide2,
    depth2,
    kernel_size,
    act,
    view_input,
):
    x = tf.keras.layers.Conv2D(
        wide1, kernel_size, padding="same", kernel_initializer=KERNEL_INITIALIZER
    )(view_input)
    x = tf.keras.layers.Activation(act)(x)
    for _ in range(depth1 - 1):
        x = tf.keras.layers.Conv2D(
            wide1, kernel_size, kernel_initializer=KERNEL_INITIALIZER
        )(x)
        x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    if wide2 * depth2 > 0:
        x = tf.keras.layers.Conv2D(
            wide2, kernel_size, padding="same", kernel_initializer=KERNEL_INITIALIZER
        )(x)
        x = tf.keras.layers.Activation(act)(x)
        for _ in range(depth2 - 1):
            x = tf.keras.layers.Conv2D(
                wide1, kernel_size, kernel_initializer=KERNEL_INITIALIZER
            )(x)
            x = tf.keras.layers.Activation(act)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    view_output = x
    return tf.keras.models.Model(inputs=[view_input], outputs=[view_output])


def mobile_net_v1(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.MobileNet(
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


def mobile_net_v2(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.MobileNetV2(
        alpha=1.0,
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


def mobile_net_v3_small(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.MobileNetV3Small(
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
        dropout_rate=0.2,
        include_preprocessing=False,
    )


def mobile_net_v3_large(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.MobileNetV3Large(
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
        dropout_rate=0.2,
        include_preprocessing=False,
    )


def nas_net_mobile(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.NASNetMobile(
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


def inception_v3(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


def resnet50v2(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


def efficient_net_b0(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


def efficient_net_v2_b0(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


def dense_net_121(**kwargs):
    view_input = kwargs.get("view_input")
    return tf.keras.applications.DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=view_input,
        pooling=None,
    )


CNN_MODEL_TABLE = {
    "classic": classic_convolutional_model,
    "mobile_v1": mobile_net_v1,
    "mobile_v2": mobile_net_v2,
    "mobile_v3s": mobile_net_v3_small,
    "mobile_v3l": mobile_net_v3_large,
    "dense": dense_net_121,
    "efficient_v1": efficient_net_b0,
    "efficient_v2": efficient_net_v2_b0,
    "resnet": resnet50v2,
    "nasnet": nas_net_mobile,
    "inception": inception_v3,
}


def generic_build_model(
    nparams,
    cnn_type,
    wide1,
    depth1,
    wide2,
    depth2,
    bottle_neck,
    wide3,
    depth3,
    kernel_size,
    act,
    optimizer,
    view_size,
    channels,
    learn_err,
):
    print("Building model with view_size:", view_size)

    # architecture
    view_input = tf.keras.layers.Input(
        shape=(view_size, view_size, channels), dtype="float32", name="view_input"
    )
    m = CNN_MODEL_TABLE[cnn_type](
        wide1=wide1,
        depth1=depth1,
        wide2=wide2,
        depth2=depth2,
        kernel_size=kernel_size,
        act=act,
        view_input=view_input,
    )
    x = m.layers[-1].output

    x = tf.keras.layers.Flatten()(x)
    view_output = tf.keras.layers.Dense(bottle_neck)(x)

    param_input = tf.keras.layers.Input(shape=[nparams], name="param_input")
    x = tf.keras.layers.concatenate([view_output, param_input])

    for _ in range(depth3):
        x = tf.keras.layers.Dense(wide3, activation=act)(x)

    if learn_err:
        out1 = tf.keras.layers.Dense(1)(x)
        out2 = tf.keras.layers.Dense(1, activation="softplus")(x)
        output = tf.keras.layers.Concatenate(name="output")([out1, out2])
        metrics = ["mae", mse, mse_var]
        loss = mse_mvar
    else:
        output = tf.keras.layers.Dense(1, name="output")(x)
        metrics = ["mae", "mse"]
        loss = "mse"
    assert len(m.inputs) == 1
    model = tf.keras.models.Model(inputs=[view_input, param_input], outputs=output)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
