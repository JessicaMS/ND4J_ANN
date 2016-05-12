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
	DataSetIterator iter;
	
	private int inputLayerSize;
	private int hiddenLayer1Size;
	//private int hiddenLayer2Size;
	private int outputLayerSize;
	
	private List<INDArray> w = new ArrayList<INDArray>();
	private List<INDArray> b = new ArrayList<INDArray>();
	
	private INDArray z2, a2, z3;//, a3, z4;
	private INDArray yHat;
	
	public NeuralNet(DataSetIterator iter, int inputLayerSize, int outputLayerSize) {
		//Seed so that each test is deterministic
		int seedValue = 12;
		this.inputLayerSize = inputLayerSize;
		this.hiddenLayer1Size = 20;
		//this.hiddenLayer2Size = 7;
		this.outputLayerSize = outputLayerSize;
		
		this.iter = iter;
		calculateScale();
		
		
		this.w.add(Nd4j.rand(new int []{inputLayerSize, hiddenLayer1Size}, seedValue));
		//this.w.set(0, this.w.get(0).div(Math.sqrt((double)inputLayerSize)));
		this.w.add(Nd4j.rand(new int []{hiddenLayer1Size, outputLayerSize}, seedValue));
		//this.w.set(1, this.w.get(1).div(Math.sqrt((double)outputLayerSize)));
		
		this.b.add(Nd4j.zeros(hiddenLayer1Size));
		describeMatrix("b1: ", this.b.get(0));
		
		this.b.add(Nd4j.zeros(outputLayerSize));
		describeMatrix("b2: ", this.b.get(1));
		//this.w.add(Nd4j.rand(new int []{hiddenLayer2Size, outputLayerSize}, seedValue));
	}
	
	public void calculateScale() {
		INDArray inputLayer = null;
		INDArray outputLayer = null;
		this.inputScale = 0;
		this.outputScale = 0;
		
		iter.reset();
		try {
			DataSet next = null;
			while(iter.hasNext()) {
				next = iter.next();
				if (next.getFeatureMatrix() == null || next.getLabels() == null)
                    break;
				inputLayer = next.getFeatureMatrix();
				outputLayer = next.getLabels();
				if (Math.abs((Double) outputLayer.maxNumber()) > this.outputScale) { 
					this.outputScale = Math.abs((Double) outputLayer.maxNumber());
				}
				if (Math.abs((Double) inputLayer.maxNumber()) > this.inputScale) {
					this.inputScale = Math.abs((Double) inputLayer.maxNumber());
				}

				//describeMatrix("inputLayer batch:", inputLayer.div(inputScale));
				//describeMatrix("outputLayer batch:", outputLayer);	
			}
			
			this.inputScale = 2.5;
			System.out.println("InputScale: " + this.inputScale);
			System.out.println("OutputScale: " + this.outputScale);	
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
	
	public INDArray output(INDArray inputLayer) {
		INDArray predictions;
		
		inputLayer = inputLayer.div(this.inputScale);
		predictions = forwardProp(inputLayer);
		return predictions;
	}

	private void describeMatrix(String description, INDArray matrix) {
		System.out.println(description);
		System.out.println(""+ matrix.columns() + " x " + matrix.rows() + " Matrix:");
		System.out.println(matrix);
	}
	
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
            y = next.getLabels();
            inputLayer = inputLayer.div(inputScale);
            y = y.div(outputScale);   //max in matrix y is scale
                       
            this.yHat = forwardProp(inputLayer);
            
            cost = MSE(y, this.yHat);
            backPropagate(cost, y, inputLayer, this.yHat);
            batchCount++;
        }
		
		return cost;
	}
	
	public INDArray forwardProp(INDArray inputLayer) {

		INDArray yHat;
		z2 = inputLayer.mmul(w.get(0));//.addRowVector(b.get(0));
		z2 = z2.addRowVector(b.get(0));
		
		a2 = sigmoid(z2);
		z3 = a2.mmul(w.get(1));//.addRowVector(b.get(1));
		//describeMatrix("z3: ", z3);
		z3 = z3.addRowVector(b.get(1));
		//describeMatrix("z3 + b2: ", z3);
		
		
		//a3 = sigmoid(z3);
		//z4 = a3.mmul(w.get(2));
		yHat = sigmoid(z3);
		
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
	
	public void updateWeights(List<INDArray> newWeights, List<INDArray> newBiases) {
		double learningFactor = 0.1;
		
		this.w.set(0, w.get(0).sub(newWeights.get(0).mul(learningFactor)));
		this.w.set(1, w.get(1).sub(newWeights.get(1).mul(learningFactor)));
		//this.w.set(2, w.get(2).sub(newWeights.get(2).mul(learningFactor)));
		
		this.b.set(0, b.get(0).subRowVector(newBiases.get(0).sum(0).mul(learningFactor)));
		
		this.b.set(1, b.get(1).subRowVector(newBiases.get(1).sum(0).mul(learningFactor)));
		
	}

	public void backPropagate(INDArray cost, INDArray y, INDArray inputs, INDArray yHat) {
	INDArray d3 = (y.sub(yHat)).mul(-1.0);
	d3 = d3.mul(sigmoidDerivative(z3));
	
	INDArray dJdW2 = a2.transpose().mmul(d3);
	INDArray d2 = d3.mmul(w.get(1).transpose()).mul(sigmoidDerivative(z2));
	INDArray dJdW1 = inputs.transpose().mmul(d2);
	
	this.updateWeights(Arrays.asList(new INDArray[]{dJdW1, dJdW2}),
						Arrays.asList(new INDArray[]{d2, d3}));
	
}
	
//	public void backPropagate(INDArray cost, INDArray y, INDArray inputs, INDArray yHat) {
//		//\nabla_a C_{MST} = (a^L - E^r)
//		INDArray d4 = (y.sub(yHat)).mul(-1.0);
//		d4 = d4.mul(sigmoidDerivative(z4));
//		
//		INDArray dJdW3 = a3.transpose().mmul(d4);
//		
//		
//		INDArray d3 = (d4.mmul(w.get(2).transpose())).mul(sigmoidDerivative(z3));
//		INDArray dJdW2 = a2.transpose().mmul(d3);
//		INDArray d2 = d3.mmul(w.get(1).transpose()).mul(sigmoidDerivative(z2));
//		INDArray dJdW1 = inputs.transpose().mmul(d2);
//		
//		this.updateWeights(Arrays.asList(new INDArray[]{dJdW1, dJdW2, dJdW3}),
//							Arrays.asList(new INDArray[]{d2, d3}));
//		
//	}
	
	
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