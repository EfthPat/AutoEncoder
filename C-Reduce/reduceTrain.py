import os
import sys
import numpy
import random
import tensorflow
from keras import layers
from tensorflow import keras
from Utils.Curve import Curve
from Utils.Parser import parse
from Utils.ArgumentParser import ArgumentParser


# Utility function to ensure the reproducibility of the results
# The function's functionality is described in the following link :
# - https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
def experimentParameters():
    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    numpy.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tensorflow.config.threading.set_inter_op_parallelism_threads(1)
    tensorflow.config.threading.set_intra_op_parallelism_threads(1)


if __name__ == '__main__':

    experimentParameters()

    argumentParser = ArgumentParser()
    argumentParser.addArgument(argument="-d", type="path", mandatory=True)

    if not argumentParser.parse(sys.argv):
        exit(1)

    path = argumentParser.getArgument("-d")

    curves = parse(path)

    kernelSize = 6
    totalLayers = 10
    latentDimension = 3
    window = 10
    epochs = 50
    batchSize = 64
    validationSplit = 0.15

    if len(curves[0]) % window != 0:
        print("Error : Curves of complexity", len(curves[0]), "can't be divided into windows of size", window)
        exit(1)

    # Normalize and split into windows each Curve
    normalisedWindowedCurves = []
    for curve in curves:
        normalisedValues = curve.normalise(curve.getValues())
        normalisedWindowedValues = numpy.reshape(normalisedValues, (-1, window, 1))
        normalisedWindowedCurves.append(Curve(curve.getID(), normalisedWindowedValues))

    # Keep 80% of the Dataset as the Training Set
    # The claim above was described in the following link :
    # - https://eclass.uoa.gr/modules/forum/viewtopic.php?course=DI352&topic=33157&forum=54779
    trainPercentage = 0.8
    trainSet, _ = Curve.splitSet(normalisedWindowedCurves, trainPercentage, asPercentage=True, shuffle=False)

    autoencoder = keras.Sequential(
        [
            layers.Input((window, 1)),
            layers.Conv1D(16, kernelSize, padding="same"),
            layers.AveragePooling1D(2, padding="same"),
            layers.Conv1D(1, kernelSize, padding="same"),
            layers.AveragePooling1D(2, padding="same"),
            layers.Conv1D(1, kernelSize, padding="same"),
            # latent-vector : output of Conv1D below
            layers.Conv1D(1, kernelSize, padding="same"),
            layers.UpSampling1D(2),
            layers.Conv1D(16, 2),
            layers.UpSampling1D(2),
            layers.Conv1D(1, 2, padding="same")
        ]
    )
    autoencoder.compile(optimizer=keras.optimizers.Adam(), loss="mae")

    # The following way of training the model is described in the following link :
    # https://eclass.uoa.gr/modules/forum/viewtopic.php?course=DI352&topic=33157&forum=54779
    # and is verified by the Keras developemnt team in the following link :
    # https://github.com/keras-team/keras/issues/4446
    for curve in trainSet:
        curveValues = curve.getValues()
        fitSummary = autoencoder.fit(curveValues, curveValues,
                                     epochs=epochs,
                                     batch_size=batchSize,
                                     validation_split=validationSplit,
                                     shuffle=True,
                                     verbose=1)
