package org.jessicams.AI.multilayerANN.NeuralNet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.dataset.DataSet;


public class NeuralNet {
	private double inputScale, outputScale;
	int trainingSetCount;
	//RecordReader rr;
	DataSetIterator iter;
	
	private int inputLayerSize;
	private int hiddenLayerSize;
	private int hiddenLayer2Size;
	private int outputLayerSize;
	
	private List<INDArray> w = new ArrayList<INDArray>();
	
	private INDArray z2, a2, z3, a3, z4;
	private INDArray yHat;
	
	public NeuralNet(DataSetIterator iter, int trainingSetCount) {
		//Seed so that each test is deterministic
		int seedValue = 12345;
		this.outputLayerSize = 1;
		this.hiddenLayerSize = 6;
		this.hiddenLayer2Size = 6;
		this.inputLayerSize = 3;
		
		this.iter = iter;
		calculateScale();
		
		this.outputScale = outputScale;
		
		this.w.add(Nd4j.rand(new int []{inputLayerSize, hiddenLayerSize}, seedValue));
		this.w.add(Nd4j.rand(new int []{hiddenLayerSize, hiddenLayer2Size}, seedValue));
		this.w.add(Nd4j.rand(new int []{hiddenLayer2Size, outputLayerSize}, seedValue));
	}
	
	public void calculateScale() {
		INDArray inputLayer = null;
		INDArray outputLayer = null;
		iter.reset();
		try {
			DataSet next = null;
			while(iter.hasNext()) {
				next = iter.next();
				if (next.getFeatureMatrix() == null || next.getLabels() == null)
                    break;
				inputLayer = next.getFeatureMatrix();
				outputLayer = buildOutputs(inputLayer);
				if ((Double) outputLayer.maxNumber() > this.outputScale) { 
					this.outputScale = (Double) outputLayer.maxNumber();
				}
				if ((Double) inputLayer.maxNumber() > this.inputScale) {
					this.inputScale = (Double) inputLayer.maxNumber();
				}
				describeMatrix("inputLayer batch:", inputLayer.div(inputScale));
				describeMatrix("outputLayer batch:", outputLayer);	
			}
			
			
		} catch (Exception e) {
			System.out.println("Next out of bounds");
			System.out.println(e.getMessage());
			e.printStackTrace();
		} finally {
				
		}
		
	}
	
	public double getOutputScale() {
		return this.outputScale;
	}
	
	public INDArray getYHat() {
		return this.yHat;
	}
	
	public void testData(double[] testingInput) {
		INDArray yHat;
		
		INDArray inputLayer = Nd4j.create(testingInput, new int[]{1, testingInput.length});
		inputLayer = inputLayer.div(this.inputScale);
		yHat = forwardProp(inputLayer);
		System.out.println("Scaled output layer:");
		System.out.println(yHat.mul(outputScale));
		System.out.println("");
	}

	private void describeMatrix(String description, INDArray matrix) {
		System.out.println(description);
		System.out.println(""+ matrix.columns() + " x " + matrix.rows() + " Matrix:");
		System.out.println(matrix);
	}
	
	public void updateWeights(List<INDArray> newWeights) {
		double learningFactor = 0.01;
		
		this.w.set(0, w.get(0).sub(newWeights.get(0).mul(learningFactor)));
		this.w.set(1, w.get(1).sub(newWeights.get(1).mul(learningFactor)));
		this.w.set(2, w.get(2).sub(newWeights.get(2).mul(learningFactor)));
	}
	
	private static INDArray buildOutputs(INDArray inputLayer) {
		double[] yArray = new double[inputLayer.rows()];
		for(int i = 0; i < inputLayer.rows(); i++) {
			double sum = 0;
			for(int j = 0; j < inputLayer.columns()-1; j++) {
				sum += inputLayer.getDouble(i, j);
			}
			yArray[i] = sum;
		}
		return Nd4j.create(yArray, new int[]{1,inputLayer.rows()});
	}
	
//	public INDArray fitSingle() {
//		//DataSetIterator iter = new RecordReaderDataSetIterator(this.rr, trainingSetCount);
//		INDArray inputLayer;
//		INDArray y;
//		INDArray cost = null;
//		iter.
//		iter.reset();	
//		try {
//			DataSet next = iter.next();
//			inputLayer = next.getFeatureMatrix();
//			y = buildOutputs(inputLayer);
//			inputLayer = inputLayer.div(inputScale);
//            y = y.div(outputScale);   //max in matrix y is scale
//                       
//            this.yHat = forwardProp(inputLayer);
//            cost = quadraticCost(y.transpose(), yHat);
//            costPrime(cost, y, inputLayer, yHat);		
//			
//		} catch (Exception e) {
//			System.out.println("Next out of bounds");
//			System.out.println(e.getMessage());
//			e.printStackTrace();
//		} finally {
//				
//		}
//		return cost;
//	}
	
	public INDArray batchTrain() {
		INDArray inputLayer;
		INDArray y;
		INDArray cost = null;
		int batchCount = 0;
		
		iter.reset();
        while (iter.hasNext()) {
            DataSet next = iter.next();
            if (next.getFeatureMatrix() == null || next.getLabels() == null)
                break;
            
            inputLayer = next.getFeatureMatrix();
            y = buildOutputs(inputLayer);
            inputLayer = inputLayer.div(inputScale);
            y = y.div(outputScale);   //max in matrix y is scale
                       
            this.yHat = forwardProp(inputLayer);
            cost = MSE(y.transpose(), this.yHat);
            costPrime(cost, y, inputLayer, this.yHat);
            batchCount++;
        }
		
		return cost;
	}
	
	public INDArray forwardProp(INDArray inputLayer) {
		INDArray yHat;
		//a1 == inputLayer in literature
		z2 = inputLayer.mmul(w.get(0));

		a2 = sigmoid(z2);
		z3 = a2.mmul(w.get(1));
		a3 = sigmoid(z3);
		z4 = a3.mmul(w.get(2));
		yHat = sigmoid(z4);
		
		return yHat;
	}
	
	public INDArray MSE(INDArray y, INDArray yHat) {
		INDArray costMatrix = y.sub(yHat);
		double totalCost;
		
		//C_{MST}(W, B, S^r, E^r) = 0.5\sum\limits_j (a^L_j - E^r_j)^2
        INDArray J = (costMatrix.mul(costMatrix)).mul(0.5);
        //this.describeMatrix("Quadratic Cost per record=", J);

		return J;
	}
	
	public void costPrime(INDArray cost, INDArray y, INDArray inputs, INDArray yHat) {
		//\nabla_a C_{MST} = (a^L - E^r)
		INDArray d4 = (y.transpose().sub(yHat)).mul(-1.0);
		d4 = d4.mul(sigmoidDerivative(z4));
		
		INDArray dJdW3 = a3.transpose().mmul(d4);
		
		//describeMatrix("sigmoidPrime2(z3): ", sigmoidPrime2(z3));
		//describeMatrix("sigmoidDerivative(z3): ", sigmoidDerivative(z3));
		
		INDArray d3 = (d4.mmul(w.get(2).transpose())).mul(sigmoidDerivative(z3));
		INDArray dJdW2 = a2.transpose().mmul(d3);
		INDArray d2 = d3.mmul(w.get(1).transpose()).mul(sigmoidDerivative(z2));
		INDArray dJdW1 = inputs.transpose().mmul(d2);
		
		this.updateWeights(Arrays.asList(new INDArray[]{dJdW1, dJdW2, dJdW3}));
		
	}
	
	
    public INDArray expi(INDArray toExp) {
        INDArray flattened = toExp.ravel();

        for (int i = 0; i < flattened.length(); i++) {
        	double element = flattened.getDouble(i);
            flattened.put(i, Nd4j.scalar(Math.exp(element)));
        }
        return flattened.reshape(toExp.shape());
    }
		
	public INDArray sigmoid(INDArray z) {
		INDArray newZ = Nd4j.getExecutioner().execAndReturn(new Sigmoid(z.dup()));
		return newZ;
	}
	
	public INDArray sigmoidDerivative(INDArray z) {
		INDArray newZ = Nd4j.getExecutioner().execAndReturn(new Sigmoid(z.dup()).derivative());
		return newZ;
	}
	
}