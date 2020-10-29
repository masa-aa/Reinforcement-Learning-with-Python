import numpy as np
from tensorflow.python import keras as K

model = K.Sequential([
    K.layers.Dense(units=4, input_shape=((2, ))),
])

weight, bias = model.layers[0].get_weights()
# 重み 2 -> 4
print("Weight shape is {}.".format(weight.shape))
# バイアス 4つの要素
print("Bias shape is {}.".format(bias.shape))

# 入力 x行2列(x:バッチサイズ)
x = np.random.rand(1, 2)
# 出力 x行4列
y = model.predict(x)
print("x is ({}) and y is ({}).".format(x.shape, y.shape))
