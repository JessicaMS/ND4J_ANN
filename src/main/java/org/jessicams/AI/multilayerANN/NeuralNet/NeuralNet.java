package org.jessicams.AI.multilayerANN.NeuralNet;

import org.nd4j.linalg.api.ndarray.INDArray;


import org.nd4j.linalg.factory.Nd4j;


public class NeuralNet {
	private int inputLayerSize;
	private int hiddenLayerSize;
	private int outputLayerSize;
	
	private INDArray w1;
	private INDArray w2;
	
	private INDArray z2, a2, z3;
	
	private INDArray yHat;
	
	public NeuralNet() {
		this.outputLayerSize = 1;
		this.hiddenLayerSize = 12;
		this.inputLayerSize = 2;

		this.w1 = Nd4j.rand(new int []{inputLayerSize, hiddenLayerSize});
		w1 = w1.sub(0.5);
		this.w2 = Nd4j.rand(new int []{hiddenLayerSize, outputLayerSize});
		w2 = w2.sub(0.5);
	}

	private void describeMatrix(String description, INDArray matrix) {
		System.out.println(description);
		System.out.println(""+ matrix.columns() + " x " + matrix.rows() + " Matrix:");
		System.out.println(matrix);
	}
	
	public void updateWeights(INDArray change1, INDArray change2) {
			
		double learningFactor = 0.9;
		
		w1 = w1.sub(change1.mul(learningFactor));
		w2 = w2.sub(change2.mul(learningFactor));
//		this.describeMatrix("w1", w1);
//		this.describeMatrix("w2", w2);
	}
	
	public INDArray forwardProp(INDArray inputLayer) {
		//a1 = inputLayer
		z2 = inputLayer.mmul(w1);
		a2 = sigmoid(z2);
		z3 = a2.mmul(w2);
		this.yHat = sigmoid(z3);
		
		return this.yHat;
	}
	
	public INDArray quadraticCost(INDArray y) {
		INDArray costMatrix = y.sub(yHat);
		double totalCost;
		
		//C_{MST}(W, B, S^r, E^r) = 0.5\sum\limits_j (a^L_j - E^r_j)^2
        INDArray J = (costMatrix.mul(costMatrix)).mul(0.5);
        this.describeMatrix("Quadratic Cost per record=", J);
        totalCost = (double)J.sumNumber().doubleValue();
        //System.out.println("Total cost: " + totalCost);
		return J;
	}
	
	public INDArray costPrime(INDArray cost, INDArray y, INDArray inputs) {
		//\nabla_a C_{MST} = (a^L - E^r)
		INDArray d3 = (y.transpose().sub(yHat)).mul(-1.0);

//		this.describeMatrix("cost function prime:", d3);
		d3 = d3.mul(sigmoidPrime(z3));
		
		INDArray dJdW2 = a2.transpose().mmul(d3); 		
//		this.describeMatrix("dJdW2 =", dJdW2);
		
		INDArray d2 = d3.mmul(w2.transpose()).mul(sigmoidPrime(z2));
//		this.describeMatrix("sigmoidPrime(z2) :", sigmoidPrime(z2));
//		this.describeMatrix("d2=", d2);
				
		INDArray dJdW1 = inputs.transpose().mmul(d2);
//		this.describeMatrix("dJdW1 =", dJdW1);
		
		this.updateWeights(dJdW1, dJdW2);
		return dJdW2;
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