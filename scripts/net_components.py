from keras.layers import Layer
import keras.backend as K

class AttLayer(Layer):
    def __init__(self, context_size):
        self._context_size = context_size
        self.supports_masking = True
        # self._linear = Dense(context_size, activation = "tanh")
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        self._W = self.add_weight(
            name = "W",
            shape = (input_shape[-1], self._context_size),
            initializer="he_normal",
            trainable=True
        )
        self._b = self.add_weight(
            name = "b",
            shape = (1, self._context_size),
            initializer="constant",
            trainable=True
        )
        self._context = self.add_weight(
            name = "context",
            shape = (self._context_size, 1),
            initializer = "he_normal",
            trainable = True
        )
        super(AttLayer, self).build(input_shape)


    def compute_mask(self, input, input_mask=None):
        return input_mask


    def call(self, input, mask = None):
        # input: (N, T, M)
        rep = K.tanh(K.dot(input, self._W) + self._b) # (N, T, C)
        score = K.squeeze(K.dot(rep, self._context), axis = -1) # (N, T)

        weight = K.exp(score)
        if mask is not None:
            weight *= K.cast(mask, K.floatx())

        weight /= K.cast(K.sum(weight, axis = 1, keepdims = True) + K.epsilon(), K.floatx())


        # weight = softmax(score, axis = -1) # (N, T)
        op = K.batch_dot(input, weight, axes = (1, 1)) # (N, M)

        return op

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



class AdditiveLayer(Layer):
    def __init__(self):
        super(AdditiveLayer, self).__init__()

    def build(self, input_shape):
        self._w = self.add_weight(
            name = "w",
            shape = (1, input_shape[-1]),
            initializer="constant",
            trainable=True
        )
        super(AdditiveLayer, self).build(input_shape)



    def call(self, input):
        return input + self._w

    def compute_output_shape(self, input_shape):
        return input_shape
