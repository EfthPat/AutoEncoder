import sys
import numpy
from Utils.Curve import Curve
from Utils.Parser import parse
from tensorflow.keras.models import load_model
from Utils.ArgumentParser import ArgumentParser

# from Utils : Curve , parse , ArgumentParser

if __name__ == '__main__':

    argumentParser = ArgumentParser()
    argumentParser.addArgument(argument="-d", type="path", mandatory=True)
    argumentParser.addArgument(argument="-q", type="path", mandatory=True)
    argumentParser.addArgument(argument="-od", type="path", mandatory=True)
    argumentParser.addArgument(argument="-oq", type="path", mandatory=True)

    if not argumentParser.parse(sys.argv):
        exit(1)

    datasetPath = argumentParser.getArgument("-d")
    querysetPath = argumentParser.getArgument("-q")
    outputDatasetPath = argumentParser.getArgument("-od")
    outputQuerysetPath = argumentParser.getArgument("-oq")

    curves = parse(datasetPath)

    if len(curves) == 0:
        exit(1)

    if datasetPath != querysetPath:
        print("Error: Dataset's path doesn't match Queryset's path")
        exit(1)

    encoder = load_model("Model")


    # Keep the encoding layers only from the autoencoder
    for i in range(4):
        encoder.pop()

    # Normalize and split each Curve into windows
    # For example, given :
    # - Curve C with values V(C) = [1,2,3,4,5,6]
    # - Window = 3
    #   the 'windowed' values of C are :
    # - V(C') = [ [ [1] , [2] , [3] ] , [ [4] , [5] , [6] ] ]
    # - where shape(C') =( ( length(C) / Window ) , Window , 1 )
    window = 10
    normalisedWindowedCurves = []
    for curve in curves:
        normalisedValues = curve.normalise(curve.getValues())
        normalisedWindowedValues = numpy.reshape(normalisedValues, (-1, window, 1))
        normalisedWindowedCurves.append(Curve(curve.getID(), normalisedWindowedValues))

    # Compress the Curves using the *encoder* , reshape and denormalize them
    compressedCurves = []
    for i in range(len(normalisedWindowedCurves)):
        normalisedWindowedValues = normalisedWindowedCurves[i].getValues()
        compressed = encoder.predict(normalisedWindowedValues)
        compressed = numpy.reshape(compressed, (-1))
        compressed = curves[i].denormalise(compressed)
        compressedCurves.append(Curve(curves[i].getID(), compressed))

    # Store the first 350 compressed Curves into the output-dataset file
    with open(outputDatasetPath, 'w') as outputFile:
        for i in range(0, 350):
            outputFile.write(compressedCurves[i].toCSV())
            outputFile.write('\n')
        outputFile.close()

    # Store the last 9 compressed Curves into the output-query file
    with open(outputQuerysetPath, 'w') as outputFile:
        for i in range(350, 359):
            outputFile.write(compressedCurves[i].toCSV())
            outputFile.write('\n')
        outputFile.close()
