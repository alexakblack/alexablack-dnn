#import tensorflow.data as data

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
test = test.cache().prefetch(bufer_size = data.AUTOTUNE)