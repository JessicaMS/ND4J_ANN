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
		int trainingSetCount =35;

		INDArray inputLayer =  buildInputs(inputLayerSize, trainingSetCount);
		INDArray y = buildOutputs(inputLayer);
		
		System.out.println("Inputs:");
		System.out.println(inputLayer);
		System.out.println("Outputs:");
		System.out.println(y);
		System.out.println("");
		
		double inputScale = (Double) inputLayer.maxNumber();
		double outputScale = (Double) y.maxNumber();
		inputLayer = inputLayer.div(inputScale);
		y = y.div(outputScale);   //max in matrix y is scale

		
		INDArray yHat, cost;

		NeuralNet myANN = new NeuralNet( inputScale, outputScale);

		for(int i = 0; i < 2000; i++) {
				yHat = myANN.forwardProp(inputLayer);
				System.out.println("Scaled output:");
				System.out.println(yHat.mul(outputScale));
				System.out.println("");

				cost = myANN.quadraticCost(y.transpose());
				myANN.costPrime(cost, y, inputLayer);		

		}

		myANN.testData(new double[]{1.0, 1.0});
		
		myANN.testData(new double[]{1.0, 2.0});		
		
		myANN.testData(new double[]{3.0, 3.0});
		
		myANN.testData(new double[]{8.0, 9.0});

	}
}
