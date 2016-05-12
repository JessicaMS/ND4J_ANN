package org.jessicams.AI.multilayerANN.NeuralNet;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuronLayer {
	private int inSize, outSize;
	private INDArray w, b;
	private Activations.Function activation;


	NeuronLayer(int inSize, int outSize, int seedValue) {
		this.setInSize(inSize);
		this.setOutSize(outSize);
		
		this.setW(Nd4j.rand(new int []{inSize, outSize}, seedValue));
		//this.w.set(0, this.w.get(0).div(Math.sqrt((double)inputLayerSize)));
		
		this.setB(Nd4j.zeros(outSize));
	}
	
	
	public Activations.Function getActivation() {
		return activation;
	}

	public void setActivation(Activations.Function activation) {
		this.activation = activation;
	}

	public int getInSize() {
		return inSize;
	}

	public void setInSize(int inSize) {
		this.inSize = inSize;
	}

	public int getOutSize() {
		return outSize;
	}

	public void setOutSize(int outSize) {
		this.outSize = outSize;
	}
	
	public void updateWeights(INDArray updates) {
		this.setW(this.getW().sub(updates));
	}
	
	public void updateBiases(INDArray updates) {
		this.setB(this.getB().sub(updates));
	}

	public INDArray getW() {
		return w;
	}

	public void setW(INDArray w) {
		this.w = w;
	}

	public INDArray getB() {
		return b;
	}

	public void setB(INDArray b) {
		this.b = b;
	}
	
	
}