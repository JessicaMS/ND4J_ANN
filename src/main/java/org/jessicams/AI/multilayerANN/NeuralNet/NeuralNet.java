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

	
	private List<NeuronLayer> Layers = new ArrayList<NeuronLayer>();
	
	private INDArray z1, a1, z2, a2;
	private INDArray yHat;
	
	public NeuralNet(DataSetIterator iter, int inputLayerSize, int outputLayerSize) {
		int seedValue = 12;  //constant seed so that each test is deterministic
		int hiddenLayer1Size = 20;
		
		this.iter = iter;
		calculateScale();
		
		Layers.add(new NeuronLayer(inputLayerSize, hiddenLayer1Size, seedValue));
		Layers.get(0).setActivation(Activations.Function.Logistic);
		Layers.add(new NeuronLayer(hiddenLayer1Size, outputLayerSize, seedValue));
		Layers.get(1).setActivation(Activations.Function.Logistic);
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
			}
			
			this.inputScale = 2.5;										//TEMPORARY STATIC NUMBER, FIX
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
		z1 = inputLayer.mmul(Layers.get(0).getW());
		z1 = z1.addRowVector(Layers.get(0).getB());
		a1 = Activations.getActivationValues(z1, Layers.get(0).getActivation());
		
		z2 = a1.mmul(Layers.get(1).getW());
		z2 = z2.addRowVector(Layers.get(1).getB());
		a2 = Activations.getActivationValues(z2, Layers.get(1).getActivation());
		
		//z4 = a3.mmul(w.get(2));
		yHat = a2;
		
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
		
		Layers.get(0).updateWeights(newWeights.get(0).mul(learningFactor));
		Layers.get(1).updateWeights(newWeights.get(1).mul(learningFactor));
		
		Layers.get(0).updateBiases(newBiases.get(0).sum(0).mul(learningFactor));
		Layers.get(1).updateBiases(newBiases.get(1).sum(0).mul(learningFactor));		
	}

	public void backPropagate(INDArray cost, INDArray y, INDArray inputs, INDArray yHat) {
	INDArray d3 = (y.sub(yHat)).mul(-1.0);
	d3 = d3.mul(Activations.getActivationDerivative(z2, Layers.get(1).getActivation()));
	INDArray dJdW2 = a1.transpose().mmul(d3);
	
	INDArray d2 = d3.mmul(Layers.get(1).getW().transpose());
	d2 = d2.mul(Activations.getActivationDerivative(z1, Layers.get(0).getActivation()));
	INDArray dJdW1 = inputs.transpose().mmul(d2);
	
	this.updateWeights(Arrays.asList(new INDArray[]{dJdW1, dJdW2}),
						Arrays.asList(new INDArray[]{d2, d3}));
	
}
	
    public INDArray expi(INDArray toExp) {
        INDArray flattened = toExp.ravel();

        for (int i = 0; i < flattened.length(); i++) {
        	double element = flattened.getDouble(i);
            flattened.put(i, Nd4j.scalar(Math.exp(element)));
        }
        return flattened.reshape(toExp.shape());
    }
	
}