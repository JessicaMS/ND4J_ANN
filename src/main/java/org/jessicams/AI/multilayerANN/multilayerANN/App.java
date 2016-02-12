package org.jessicams.AI.multilayerANN.multilayerANN;

import org.jessicams.AI.multilayerANN.NeuralNet.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Hello world!
 *
 */
public class App 
{
	private static INDArray buildInputs(int inputLayerSize, int trainingSetCount) {
		double[] inputLayerArray = new double[inputLayerSize*trainingSetCount];
		for(int i = 0; i < inputLayerSize; i++){
			for(int j = 0; j < trainingSetCount; j++) {
				inputLayerArray[inputLayerSize*j+i] = (j/4.0)+ 1.0 + (i*j % 4);
			}
		}

		return Nd4j.create(inputLayerArray, new int[]{trainingSetCount, inputLayerSize}); 
	}

	private static INDArray buildOutputs(INDArray inputLayer) {
		double[] yArray = new double[inputLayer.rows()];
		for(int i = 0; i < inputLayer.rows(); i++) {
			double sum = 1;
			for(int j = 0; j < inputLayer.columns(); j++) {
				sum += inputLayer.getDouble(i, j);
			}
			yArray[i] = sum;
		}

		return Nd4j.create(yArray, new int[]{1,inputLayer.rows()});
	}

	public static void main( String[] args )
	{
		int inputLayerSize = 2;
		int trainingSetCount =44;

		INDArray inputLayer =  buildInputs(inputLayerSize, trainingSetCount);
		INDArray y = buildOutputs(inputLayer);
		
		System.out.println("Inputs:");
		System.out.println(inputLayer);
		System.out.println("Outputs:");
		System.out.println(y);

		System.out.println("");
		double inputScale = (Double) inputLayer.maxNumber();
		double outputScale = (Double) y.maxNumber();
		System.out.println(inputScale + " | " + outputScale);
		inputLayer = inputLayer.div(inputScale);
		y = y.div(outputScale);   //max in matrix y is scale

		
		INDArray yHat, cost;

		NeuralNet myANN = new NeuralNet();

		for(int i = 0; i < 70; i++) {
				yHat = myANN.forwardProp(inputLayer);
				System.out.println("Scaled output:");
				System.out.println(yHat.mul(outputScale));
				System.out.println("");

				cost = myANN.quadraticCost(y.transpose());
				myANN.costPrime(cost, y, inputLayer);		

		}

		INDArray inputLayer2 = Nd4j.create(new double[]{1.0, 2.0}, new int[]{1, inputLayerSize});
		inputLayer2 = inputLayer2.div(inputScale);

		yHat = myANN.forwardProp(inputLayer2);
		System.out.println("Scaled output layer 2:");
		System.out.println(yHat.mul(outputScale));
		System.out.println("");
		
		INDArray inputLayer3 = Nd4j.create(new double[]{2.0, 4.0}, new int[]{1, inputLayerSize});
		inputLayer3 = inputLayer3.div(inputScale);

		yHat = myANN.forwardProp(inputLayer3);
		System.out.println("Scaled output layer 3:");
		System.out.println(yHat.mul(outputScale));
		System.out.println("");
		
		INDArray inputLayer4 = Nd4j.create(new double[]{3.0, 3.0}, new int[]{1, inputLayerSize});
		inputLayer4 = inputLayer4.div(inputScale);

		yHat = myANN.forwardProp(inputLayer4);
		System.out.println("Scaled output layer 4:");
		System.out.println(yHat.mul(outputScale));
		System.out.println("");

	}
}
