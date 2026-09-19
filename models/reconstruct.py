from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    UpSampling2D,
    LeakyReLU,
    BatchNormalization,
    Activation,
    Concatenate,
)
from tensorflow.python.framework.ops import disable_eager_execution
from models.PartialConv import PConv2D

disable_eager_execution()


class PCModel(object):
    """Partial-convolution U-Net that maps an OCT fundus image to an RNFLT map."""

    def __init__(self, img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def build_pconv_unet(self, train_bn=True):
        inputs_img = Input((self.img_rows, self.img_cols, 3), name='inputs_img')
        inputs_mask = Input((self.img_rows, self.img_cols, 3), name='inputs_mask')

        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(
                kernel_size=kernel_size,
                n_channels=3,
                mono=False,
                filters=filters,
                strides=2,
                padding='same',
            )([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN' + str(encoder_layer.counter))(
                    conv, training=train_bn
                )
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask

        encoder_layer.counter = 0

        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
        e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3)
        e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 512, 3)

        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_img = UpSampling2D(size=(2, 2))(img_in)
            up_mask = UpSampling2D(size=(2, 2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv, up_img])
            concat_mask = Concatenate(axis=3)([e_mask, up_mask])
            conv, mask = PConv2D(
                kernel_size=kernel_size, padding='same', filters=filters
            )([concat_img, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask

        d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 512, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 512, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3)
        d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
        d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
        d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
        d_conv16, d_mask16 = decoder_layer(
            d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False
        )
        outputs = Conv2D(1, 1, name='pred')(d_conv16)

        self.model = Model(
            inputs=[inputs_img, inputs_mask],
            outputs=outputs,
            name='reconstruct_model',
        )
        return self.model

    def load(self, weight_path, train_bn=False, lr=0.00001):
        """Build the U-Net, compile it, and load pretrained weights."""
        from tensorflow.keras.optimizers import Adam

        self.build_pconv_unet(train_bn=train_bn)
        try:
            optimizer = Adam(lr=lr)
        except TypeError:
            optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        self.model.load_weights(weight_path)
        return self.model
