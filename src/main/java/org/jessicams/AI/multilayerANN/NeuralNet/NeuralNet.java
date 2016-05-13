package org.jessicams.AI.multilayerANN.NeuralNet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;

/**
 * 
 * @author Jessica Seibert
 *
 */
public class NeuralNet {
	private double inputScale, outputScale;
	DataSetIterator iter;
	private int layerCount;
	private double learningFactor;
	
	private List<NeuronLayer> Layers = new ArrayList<NeuronLayer>();
	private List<INDArray> z = new ArrayList<INDArray>();
	private List<INDArray> a = new ArrayList<INDArray>();
	
	private INDArray yHat;
	
	public NeuralNet(DataSetIterator iter, int inputLayerSize, int outputLayerSize) {
		int seedValue = 12;  //constant seed so that each test is deterministic
		int hiddenLayer1Size = 5;
		int hiddenLayer2Size = 5;
		int hiddenLayer3Size = 5;
		
		this.learningFactor = 0.05;
		this.iter = iter;
		calculateScale();
		
		Layers.add(new NeuronLayer(inputLayerSize, hiddenLayer1Size, seedValue));
		Layers.get(0).setActivation(Activations.Function.ELU);
		Layers.add(new NeuronLayer(hiddenLayer1Size, hiddenLayer2Size, seedValue));
		Layers.get(1).setActivation(Activations.Function.ELU);
		Layers.add(new NeuronLayer(hiddenLayer2Size, hiddenLayer3Size, seedValue));
		Layers.get(2).setActivation(Activations.Function.ELU);
		Layers.add(new NeuronLayer(hiddenLayer3Size, outputLayerSize, seedValue));
		Layers.get(3).setActivation(Activations.Function.Logistic);
		this.layerCount = Layers.size();
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
		
		a.clear();
		z.clear();
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
                       
            a.clear();
            z.clear();
            this.yHat = forwardProp(inputLayer);
            
            cost = MSE(y, this.yHat);
            backPropagate(y, this.yHat);
            batchCount++;
        }
		
		return cost;
	}
	
	
	public INDArray forwardProp(INDArray inputLayer) {
		INDArray yHat;
		a.add(inputLayer);
		
		for(int i = 0; i < Layers.size(); i++) {
			z.add(a.get(i).mmul(Layers.get(i).getW()));
			z.set(i, z.get(i).addRowVector(Layers.get(i).getB()));
			a.add(Activations.getActivationValues(z.get(i), Layers.get(i).getActivation()));
		}
		
		yHat = a.get(Layers.size());
		
		return yHat;
	}

	
	public INDArray MSE(INDArray y, INDArray yHat) {
		INDArray costMatrix = y.sub(yHat);
        INDArray J = (costMatrix.mul(costMatrix)).mul(0.5);
		return J;
	}
	
	public INDArray MSEPrime(INDArray y, INDArray yHat) {
		return (y.sub(yHat)).mul(-1.0);
	}
	
	
	
	public void updateWeights(List<INDArray> newWeights, List<INDArray> newBiases) {
		for(int i = 0; i < Layers.size(); i++) {
			Layers.get(i).updateWeights(newWeights.get(i).mul(learningFactor));		
			Layers.get(i).updateBiases(newBiases.get(i).sum(0).mul(learningFactor));
		}
	}

	public void backPropagate(INDArray y, INDArray yHat) {
	List<INDArray> dJdW = new ArrayList<INDArray>();
	List<INDArray> d = new ArrayList<INDArray>();
	int bpLayer = this.layerCount -1;
	d.add(MSEPrime(y, yHat));  //J prime AKA cost prime
	
	int i;
	for(i = 0; i < bpLayer; i++) {
		d.set(i, d.get(i).mul(Activations.getActivationDerivative(z.get(bpLayer-i), Layers.get(bpLayer-i).getActivation())));
		dJdW.add(a.get(bpLayer-i).transpose().mmul(d.get(i)));
		d.add(d.get(i).mmul(Layers.get(bpLayer-i).getW().transpose()));	
	}
	d.set(i, d.get(i).mul(Activations.getActivationDerivative(z.get(bpLayer-i), Layers.get(bpLayer-i).getActivation())));
	dJdW.add(a.get(bpLayer-i).transpose().mmul(d.get(i)));
	
	Collections.reverse(dJdW);
	Collections.reverse(d);
	this.updateWeights(dJdW, d);
	
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