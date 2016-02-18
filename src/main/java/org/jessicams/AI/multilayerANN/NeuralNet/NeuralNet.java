package org.jessicams.AI.multilayerANN.NeuralNet;

import org.nd4j.linalg.api.ndarray.INDArray;


import org.nd4j.linalg.factory.Nd4j;


public class NeuralNet {
	private double inputScale, outputScale;
	
	private int inputLayerSize;
	private int hiddenLayerSize;
	private int hiddenLayer2Size;
	private int outputLayerSize;
	
	private INDArray w[];
	private INDArray w1;
	private INDArray w2;
	private INDArray w3;
	
	private INDArray z2, a2, z3, a3, z4;
	
	private INDArray yHat;
	
	public NeuralNet(double inputScale, double outputScale) {
		this.outputLayerSize = 1;
		this.hiddenLayerSize = 11;
		this.hiddenLayer2Size = 11;
		this.inputLayerSize = 2;
		
		this.inputScale = inputScale;
		this.outputScale = outputScale;

		
		this.w1 = Nd4j.rand(new int []{inputLayerSize, hiddenLayerSize});
		this.w2 = Nd4j.rand(new int []{hiddenLayerSize, hiddenLayer2Size});
		this.w3 = Nd4j.rand(new int[]{hiddenLayer2Size, outputLayerSize});
	}
	
	public void testData(double[] testingInput) {
		INDArray yHat;
		
		INDArray inputLayer = Nd4j.create(testingInput, new int[]{1, testingInput.length});
		inputLayer = inputLayer.div(inputScale);
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
	
	public void updateWeights(INDArray change1, INDArray change2, INDArray change3) {
			
		double learningFactor = 0.5;
		
		
		w1 = w1.sub(change1.mul(learningFactor));
		w2 = w2.sub(change2.mul(learningFactor));
		w3 = w3.sub(change3.mul(learningFactor));
	}
	
	public INDArray forwardProp(INDArray inputLayer) {
		//a1 = inputLayer
		z2 = inputLayer.mmul(w1);
		a2 = sigmoid(z2);
		z3 = a2.mmul(w2);
		a3 = sigmoid(z3);
		z4 = a3.mmul(w3);
		this.yHat = sigmoid(z4);
		
		return this.yHat;
	}
	
	public INDArray quadraticCost(INDArray y) {
		INDArray costMatrix = y.sub(yHat);
		double totalCost;
		
		//C_{MST}(W, B, S^r, E^r) = 0.5\sum\limits_j (a^L_j - E^r_j)^2
        INDArray J = (costMatrix.mul(costMatrix)).mul(0.5);
        //this.describeMatrix("Quadratic Cost per record=", J);
        totalCost = (double)J.sumNumber().doubleValue();
        System.out.println("Total cost: " + totalCost);
		return J;
	}
	
	public void costPrime(INDArray cost, INDArray y, INDArray inputs) {
		//\nabla_a C_{MST} = (a^L - E^r)
		INDArray d4 = (y.transpose().sub(yHat)).mul(-1.0);
		d4 = d4.mul(sigmoidPrime(z4));
		
		INDArray dJdW3 = a3.transpose().mmul(d4);
//		this.describeMatrix("dJdW3 = ", dJdW3);
//		this.describeMatrix("z3: ",  z3);
//		this.describeMatrix("a2: " ,  a2);
		//this.describeMatrix("w3.mul(sigmoidPrime(z3))  " ,w3.mul(sigmoidPrime(z3)) );
		
		INDArray d3 = (d4.mmul(w3.transpose())).mul(sigmoidPrime(z3));
		
		INDArray dJdW2 = a2.transpose().mmul(d3);
		
		INDArray d2 = d3.mmul(w2.transpose()).mul(sigmoidPrime(z2));
				
		INDArray dJdW1 = inputs.transpose().mmul(d2);
		
		this.updateWeights(dJdW1, dJdW2, dJdW3);
		
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
		INDArray sigmoid;
		
		sigmoid = this.expi(z.mul(-1.0)).add(1.0);
		sigmoid = sigmoid.rdiv(1.0);
		return sigmoid;
	}
	
	public INDArray sigmoidPrime(INDArray z) {
		//exp(-z)/((1+exp(-z))**2)
		INDArray ez = expi(z);
		ez = ez.div( (ez.add(1.0)).mul( ez.add(1.0) ) );
		
		return ez;
	}
	
	public INDArray sigmoidPrime2(INDArray z) {
		return z.mul(z.rsub(1.0));
	}
	
	
}