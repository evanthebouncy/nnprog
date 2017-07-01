import tensorflow as tf


class Architecture():
    def __init__(self,
                 numberOfFilters,
                 kernelSizes,
                 poolSizes,
                 poolStrides):
        self.poolStrides = poolStrides
        self.numberOfFilters = numberOfFilters
        self.kernelSizes = kernelSizes
        self.poolSizes = poolSizes

    def makeModel(self,imageInput):

        numberOfFilters = self.numberOfFilters
        kernelSizes = self.kernelSizes
        poolSizes = self.poolSizes
        poolStrides = self.poolStrides
        nextInput = imageInput
        for filterCount,kernelSize,poolSize,poolStride in zip(numberOfFilters,kernelSizes,poolSizes,poolStrides):
            c1 = tf.layers.conv2d(inputs = nextInput,
                                  filters = filterCount,
                                  kernel_size = 2*[kernelSize],
                                  padding = "same",
                                  activation = tf.nn.relu,
                                  strides = 1)

            c1 = tf.layers.max_pooling2d(inputs = c1,
                                         pool_size = poolSize,
                                         strides = poolStride,
                                         padding = "same")
            print "Convolution output:",c1
            nextInput = c1
        return nextInput

architectures = {}
architectures["v1_2D"] = Architecture(numberOfFilters = [4],
                                      kernelSizes = [8,8],
                                      poolSizes = [8,4],
                                      poolStrides = [4,4])
