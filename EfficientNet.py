import os
import math
import string
import collections
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.python.platform.tf_logging import warning

BASE_WEIGHTS_PATH = 'https://storage.googleapis.com/keras-applications/'
WEIGHTS_HASHES = {
    'b0': ('902e53a9f72be733fc0bcb005b3ebbac',
           '50bc09e76180e00e4465e1a485ddc09d'),
    'b1': ('1d254153d4ab51201f1646940f018540',
           '74c4e6b3e1f6a1eea24c589628592432'),
    'b2': ('b15cce36ff4dcbd00b6dd88e7857a6ad',
           '111f8e2ac8aa800a7a99e3239f7bfb39'),
    'b3': ('ffd1fdc53d0ce67064dc6a9c7960ede0',
           'af6d107764bb5b1abb91932881670226'),
    'b4': ('18c95ad55216b8f92d7e70b3a046e2fc',
           'ebc24e6d6c33eaebbd558eafbeedf1ba'),
    'b5': ('ace28f2a6363774853a83a0b21b9421a',
           '38879255a25d3c92d5e44e04ae6cec6f'),
    'b6': ('165f6e37dce68623721b423839de8be5',
           '9ecce42647a20130c1f39a5d4cb75743'),
    'b7': ('8c03f828fec3ef71311cd463b6759d99',
           'cbcfe4450ddf6f3ad90b1b398090fe4a'),
}

IMAGENET_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')
IMAGENET_WEIGHTS_HASHES = {
    'efficientnet-b0': ('163292582f1c6eaca8e7dc7b51b01c61'
                        '5b0dbc0039699b4dcd0b975cc21533dc',
                        'c1421ad80a9fc67c2cc4000f666aa507'
                        '89ce39eedb4e06d531b0c593890ccff3'),
    'efficientnet-b1': ('d0a71ddf51ef7a0ca425bab32b7fa7f1'
                        '6043ee598ecee73fc674d9560c8f09b0',
                        '75de265d03ac52fa74f2f510455ba64f'
                        '9c7c5fd96dc923cd4bfefa3d680c4b68'),
    'efficientnet-b2': ('bb5451507a6418a574534aa76a91b106'
                        'f6b605f3b5dde0b21055694319853086',
                        '433b60584fafba1ea3de07443b74cfd3'
                        '2ce004a012020b07ef69e22ba8669333'),
    'efficientnet-b3': ('03f1fba367f070bd2545f081cfa7f3e7'
                        '6f5e1aa3b6f4db700f00552901e75ab9',
                        'c5d42eb6cfae8567b418ad3845cfd63a'
                        'a48b87f1bd5df8658a49375a9f3135c7'),
    'efficientnet-b4': ('98852de93f74d9833c8640474b2c698d'
                        'b45ec60690c75b3bacb1845e907bf94f',
                        '7942c1407ff1feb34113995864970cd4'
                        'd9d91ea64877e8d9c38b6c1e0767c411'),
    'efficientnet-b5': ('30172f1d45f9b8a41352d4219bf930ee'
                        '3339025fd26ab314a817ba8918fefc7d',
                        '9d197bc2bfe29165c10a2af8c2ebc675'
                        '07f5d70456f09e584c71b822941b1952'),
    'efficientnet-b6': ('f5270466747753485a082092ac9939ca'
                        'a546eb3f09edca6d6fff842cad938720',
                        '1d0923bb038f2f8060faaf0a0449db4b'
                        '96549a881747b7c7678724ac79f427ed'),
    'efficientnet-b7': ('876a41319980638fa597acbbf956a82d'
                        '10819531ff2dcb1a52277f10c7aefa1a',
                        '60b56ff3a8daccc8d96edfd40b204c11'
                        '3e51748da657afd58034d54d3cec2bac')
}

NS_WEIGHTS_PATH = 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/'
NS_WEIGHTS_HASHES = {
    'efficientnet-b0': ('5e376ca93bc6ba60f5245d13d44e4323', 'a5b48ae7547fc990c7e4f3951230290d'),
    'efficientnet-b1': ('79d29151fdaec95ac78e1ca97fc09634', '4d35baa41ca36f175506a33918f7e334'),
    'efficientnet-b2': ('8c643222ffb73a2bfdbdf90f2cde01af', 'e496e531f41242598288ff3a4b4199f9'),
    'efficientnet-b3': ('3b29e32602dad75d1f575d9ded00f930', '47da5b154de1372b557a65795d3e6135'),
    'efficientnet-b4': ('c000bfa03bf3c93557851b4e1fe18f51', '47c10902a4949eec589ab92fe1c35ed8'),
    'efficientnet-b5': ('8a920cd4ee793f53c251a1ecd3a5cee6', '4d53ef3544d4114e2d8080d6d777a74c'),
    'efficientnet-b6': ('cc69df409516ab57e30e51016326853e', '71f96d7e15d9f891f3729b4f4e701f77'),
    'efficientnet-b7': ('1ac825752cbc26901c8952e030ae4dd9', 'e112b00c464fe929b821edbb35d1af55')
}

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def get_swish():
    def swish(x):
        return tf.nn.swish(x)
    return swish

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))

def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='',):
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x=layers.Conv2D(filters,1,padding="same",use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=prefix+"expand_conv")(inputs)
        x=layers.BatchNormalization(axis=bn_axis,name=prefix+'expand_bn')(x)
        x=layers.Activation(activation,name=prefix+'expand_activation')(x)
    else:
        x=inputs
    # Depthwise Convolution
    x=layers.DepthwiseConv2D(block_args.kernel_size,strides=block_args.strides,
                            padding="same",use_bias=False,
                            depthwise_initializer=CONV_KERNEL_INITIALIZER,
                            name=prefix+"dwconv")(x)
    x=layers.BatchNormalization(axis=bn_axis,name=prefix+"bn")(x)
    x=layers.Activation(activation,name=prefix+'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters=max(1,int(block_args.input_filters * block_args.se_ratio))
        se_tensor=layers.GlobalAveragePooling2D(name=prefix+'sesqueeze')(x)

        target_shape = (1,1,filters) if tf.keras.backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters,1,activation=activation,
                                    padding='same',use_bias=True,
                                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                                    name=prefix+'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters,1,activation='sigmoid',padding='same',use_bias=True,
                                    kernel_initializer=CONV_KERNEL_INITIALIZER,name=prefix+'se_expand')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x=layers.Conv2D(block_args.output_filters,1,padding='same',use_bias=False,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,name=prefix + 'project_conv')(x)
    x=layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(s == 1 for s in block_args.strides) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x=layers.Dropout(drop_rate,noise_shape=(None,1,1,1),name=prefix+'drop')(x)
        x=layers.add([x,inputs],name=prefix+'add')
    return x

def _obtain_input_shape	(input_shape,default_size,min_size,data_format,require_flatten,weights = None ):
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if tf.keras.backend.image_data_format == 'channels_first':
               if input_shape[0] not in {1, 3}:
                warning.warn(
                       'This model usually expects 1 or 3 input channels. '
                       'However, it was passed an input_shape with ' +
                       str(input_shape[0]) + ' input channels.')
               default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warning.warn(
                       'This model usually expects 1 or 3 input channels. '
                       'However, it was passed an input_shape with ' +
                        str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if tf.keras.backend.image_data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                'and loading `imagenet` weights, '
                                '`input_shape` should be ' +
                                str(default_shape) + '.')
        return default_shape
    if input_shape:
        if tf.keras.backend.image_data_format() == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got `input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' + str(min_size) + 'x' + str(min_size) + '; got ''`input_shape=' + str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got ''`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or(input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' + str(min_size) + 'x' + str(min_size) + '; got ' '`input_shape=' + str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if tf.keras.backend.image_data_format() == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
        if require_flatten:
            if None in input_shape:
                raise ValueError('If `include_top` is True, ''you should specify a static `input_shape`. ''Got `input_shape=' + str(input_shape) + '`')
        return input_shape

def EfficientNet(width_coefficient,depth_coefficient,default_resolution,dropout_rate=0.2,drop_connect_rate=0.2,
                depth_divisor=8,blocks_args=DEFAULT_BLOCKS_ARGS,model_name='efficientnet',include_top=True,
                weights='imagenet',input_tensor=None,input_shape=None,pooling=None,classes=1000,**kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected layer at the top of the network.
        weights: one of `None` (random initialization),'imagenet' (pre-training on ImageNet),or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor(i.e. output of `layers.Input()`) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if `include_top` is False.It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,or invalid input shape.
    """
    if not (weights in {'imagenet', 'noisy-student', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either ''`None` (random initialization), `imagenet` ''(pre-training on ImageNet), ''or the path to the weights file to be loaded.')
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`' ' as true, `classes` should be 1000')

    input_shape = _obtain_input_shape(input_shape,default_size=default_resolution,min_size=32,
                                        data_format=tf.keras.backend.image_data_format(),
                                        require_flatten=include_top,weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        from tensorflow.python.keras.backend import is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input =layers.Input(tensor=input_tensor,shape=input_shape)
        else:
            img_input = input_tensor
    
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    activation = get_swish()

    x=img_input
    x=layers.Conv2D(round_filters(32,width_coefficient,depth_divisor),3,strides=(2,2),padding='same',
                    use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,name='stem_conv')(x)
    x=layers.BatchNormalization(axis=bn_axis,name="stem_bn")(x)
    x=layers.Activation(activation,name='stem_activation')(x)

    num_blocks_total=sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(input_filters=round_filters(block_args.input_filters,width_coefficient, depth_divisor),
                                        output_filters=round_filters(block_args.output_filters,width_coefficient, depth_divisor),
                                        num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))
        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x,block_args,activation=activation,drop_rate=drop_rate,prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in range(block_args.num_repeat - 1):
                drop_rate=drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1,string.ascii_lowercase[bidx + 1])
                x = mb_conv_block(x, block_args,activation=activation,drop_rate=drop_rate,prefix=block_prefix)
                block_num +=1

    x=layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,padding='same',
                    use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,name='top_conv')(x)
    x=layers.BatchNormalization(axis=bn_axis,name='top_bn')(x)
    x=layers.Activation(activation,name='top_activation')(x)
    if include_top:
        x=layers.GlobalAveragePooling2D(name='avg_poll')(x)
        if dropout_rate and drop_rate >0 :
            x=layers.Dropout(drop_rate,name='top_dropout')(x)
            x=layers.Dense(classes,activation='softmax',kernel_initializer=DENSE_KERNEL_INITIALIZER,name='probx')(x)
    else:
        if pooling == 'avg':
            x=layers.GlobalAveragePooling2D(name='avg_poll')(x)
        elif pooling == 'max':
            x=layers.GlobalAveragePooling2D(name='max_pool')(x)
    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs=img_input
    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
        weights_path=tf.keras.utils.get_file(file_name,IMAGENET_WEIGHTS_PATH + file_name, cache_subdir='models',file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights == 'noisy-students':
        if include_top:
            file_name = "{}_{}.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = "{}_{}_notop.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][1]
        weights_path=tf.keras.utils.get_file(file_name,NS_WEIGHTS_PATH + file_name,cache_subdir='models',file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model

def EfficientNetB0(include_top=True,weights='imagenet',input_tensor=None,
                    input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(1.0, 1.0, 224, 0.2,model_name='efficientnet-b0',
                    include_top=include_top, weights=weights,input_tensor=input_tensor,
                    input_shape=input_shape,pooling=pooling, classes=classes, **kwargs)

def EfficientNetB1(include_top=True,weights='imagenet',input_tensor=None,
                    input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(1.0, 1.1, 240, 0.2,model_name='efficientnet-b1',
                    include_top=include_top, weights=weights,
                    input_tensor=input_tensor, input_shape=input_shape,
                    pooling=pooling, classes=classes,**kwargs)

def EfficientNetB2(include_top=True,weights='imagenet',input_tensor=None,
                    input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(1.1, 1.2, 260, 0.3,model_name='efficientnet-b2',
                    include_top=include_top, weights=weights,
                    input_tensor=input_tensor, input_shape=input_shape,
                    pooling=pooling, classes=classes,**kwargs)

def EfficientNetB3(include_top=True,weights='imagenet',input_tensor=None,
                   input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(1.2, 1.4, 300, 0.3, model_name='efficientnet-b3',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,**kwargs
    )

def EfficientNetB4(include_top=True,weights='imagenet',input_tensor=None,
                    input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(1.4, 1.8, 380, 0.4,model_name='efficientnet-b4',
                    include_top=include_top, weights=weights,
                    input_tensor=input_tensor, input_shape=input_shape,
                    pooling=pooling, classes=classes,**kwargs)

def EfficientNetB5(include_top=True,weights='imagenet',input_tensor=None,
                    input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(1.6, 2.2, 456, 0.4,model_name='efficientnet-b5',
                    include_top=include_top, weights=weights,
                    input_tensor=input_tensor, input_shape=input_shape,
                    pooling=pooling, classes=classes,**kwargs)

def EfficientNetB6(include_top=True,weights='imagenet',input_tensor=None,
                input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(
                1.8, 2.6, 528, 0.5,model_name='efficientnet-b6',
                include_top=include_top, weights=weights,
                input_tensor=input_tensor, input_shape=input_shape,
                pooling=pooling, classes=classes,**kwargs)

def EfficientNetB7(include_top=True,weights='imagenet',input_tensor=None,
                input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(
                2.0, 3.1, 600, 0.5,model_name='efficientnet-b7',
                include_top=include_top, weights=weights,
                input_tensor=input_tensor, input_shape=input_shape,
                pooling=pooling, classes=classes,**kwargs)

def EfficientNetL2(include_top=True,weights='imagenet',input_tensor=None,
                input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(4.3, 5.3, 800, 0.5,model_name='efficientnet-l2',
                include_top=include_top, weights=weights,
                input_tensor=input_tensor, input_shape=input_shape,
                pooling=pooling, classes=classes,**kwargs)
