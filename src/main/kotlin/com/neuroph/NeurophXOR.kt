package com.neuroph

import org.neuroph.core.Layer
import org.neuroph.core.NeuralNetwork
import org.neuroph.core.Neuron
import org.neuroph.core.data.DataSet
import org.neuroph.core.data.DataSetRow
import org.neuroph.core.learning.LearningRule
import org.neuroph.nnet.learning.BackPropagation
import org.neuroph.util.ConnectionFactory
import org.neuroph.util.NeuralNetworkType

fun assembleNeuralNetwork(): NeuralNetwork<LearningRule> {
    val inputLayer = Layer()
    inputLayer.addNeuron(Neuron())
    inputLayer.addNeuron(Neuron())

    val hiddenLayerOne = Layer()

    hiddenLayerOne.addNeuron(Neuron())
    hiddenLayerOne.addNeuron(Neuron())
    hiddenLayerOne.addNeuron(Neuron())
    hiddenLayerOne.addNeuron(Neuron())

    val hiddenLayerTwo = Layer()

    hiddenLayerTwo.addNeuron(Neuron())
    hiddenLayerTwo.addNeuron(Neuron())
    hiddenLayerTwo.addNeuron(Neuron())
    hiddenLayerTwo.addNeuron(Neuron())

    val outputLayer = Layer()

    outputLayer.addNeuron(Neuron())

    val neuralNetwork: NeuralNetwork<LearningRule> = NeuralNetwork()

    neuralNetwork.addLayer(0, inputLayer)
    neuralNetwork.addLayer(1, hiddenLayerOne)

    ConnectionFactory.fullConnect(neuralNetwork.getLayerAt(0), neuralNetwork.getLayerAt(1))

    neuralNetwork.addLayer(2, hiddenLayerTwo)

    ConnectionFactory.fullConnect(neuralNetwork.getLayerAt(1), neuralNetwork.getLayerAt(2))

    neuralNetwork.addLayer(3, outputLayer)

    ConnectionFactory.fullConnect(neuralNetwork.getLayerAt(2), neuralNetwork.getLayerAt(3))
    ConnectionFactory.fullConnect(neuralNetwork.getLayerAt(0), neuralNetwork.getLayerAt(neuralNetwork.layersCount - 1), false)

    neuralNetwork.inputNeurons = inputLayer.neurons
    neuralNetwork.outputNeurons = outputLayer.neurons
    neuralNetwork.networkType = NeuralNetworkType.MULTI_LAYER_PERCEPTRON
    return neuralNetwork
}

fun trainNeuralNetwork(neuralNetwork: NeuralNetwork<LearningRule>): NeuralNetwork<LearningRule> {
    val inputSize = 2
    val outputSize = 1
    val ds = DataSet(inputSize, outputSize)
    val rOne = DataSetRow(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))
    ds.addRow(rOne)
    val rTwo = DataSetRow(doubleArrayOf(1.0, 1.0), doubleArrayOf(0.0))
    ds.addRow(rTwo)
    val rThree = DataSetRow(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0))
    ds.addRow(rThree)
    val rFour = DataSetRow(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0))
    ds.addRow(rFour)

    val backPropagation = BackPropagation()

    backPropagation.maxIterations = 1000

    neuralNetwork.learn(ds, backPropagation)

    return neuralNetwork
}