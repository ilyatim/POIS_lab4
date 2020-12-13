import com.neuroph.assembleNeuralNetwork
import com.neuroph.trainNeuralNetwork
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.neuroph.core.NeuralNetwork
import org.neuroph.core.learning.LearningRule


class XORIntegrationTest {

    private var neuralNetwork: NeuralNetwork<LearningRule>? = null
    private fun print(input: String, output: Double, actual: Double) {
        println("Testing: $input Expected: $actual Result: $output")
    }

    @Before
    fun networkInit() {
        neuralNetwork = trainNeuralNetwork(assembleNeuralNetwork())
    }

    @Test
    fun leftTest() {
        neuralNetwork?.setInput(0.0, 1.0)
        neuralNetwork?.calculate()
        neuralNetwork?.output?.get(0)?.let { print("0, 1", it, 1.0) }
        neuralNetwork?.output?.get(0)?.let { assertEquals(it, 1.0, 0.0) }
    }

    @Test
    fun rightTest() {
        neuralNetwork?.setInput(1.0, 0.0)
        neuralNetwork?.calculate()
        neuralNetwork?.output?.get(0)?.let { print("1, 0", it, 1.0) }
        neuralNetwork?.output?.get(0)?.let { assertEquals(it, 1.0, 0.0) }
    }

    @Test
    fun bothFalseTest() {
        neuralNetwork?.setInput(0.0, 0.0)
        neuralNetwork?.calculate()
        neuralNetwork?.output?.get(0)?.let { print("0, 0", it, 0.0) }
        assertEquals(neuralNetwork?.output!![0], 0.0, 0.0)
    }

    @Test
    fun bothTrueTest() {
        neuralNetwork?.setInput(1.0, 1.0)
        neuralNetwork?.calculate()
        neuralNetwork?.output?.get(0)?.let { print("1, 1", it, 0.0) }
        neuralNetwork?.output?.get(0)?.let { assertEquals(it, 0.0, 0.0) }
    }

    @After
    fun networkClose() {
        neuralNetwork = null
    }
}