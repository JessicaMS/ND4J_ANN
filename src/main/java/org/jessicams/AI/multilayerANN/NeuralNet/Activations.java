package org.jessicams.AI.multilayerANN.NeuralNet;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.factory.Nd4j;

public class Activations {
    public enum Function {Tanh,
				    	Logistic,  
				    	ReLU, 
				    	LeakyReLU, 
				    	ELU};
	
	//Return transform of passed matrix
    public static INDArray getActivationValues(INDArray x, Function function){
        switch (function){
            case Tanh:
                return Nd4j.getExecutioner().execAndReturn(new Tanh(x.dup()));
            case Logistic:
            	return Nd4j.getExecutioner().execAndReturn(new Sigmoid(x.dup()));
            case ReLU:
            	return Nd4j.getExecutioner().execAndReturn(new RectifedLinear(x.dup()));
            case LeakyReLU:
            	return Nd4j.getExecutioner().execAndReturn(new LeakyReLU(x.dup()));
            case ELU:
            	return Nd4j.getExecutioner().execAndReturn(new ELU(x.dup()));
            default:
                throw new RuntimeException();
        }
    }
    
    public static INDArray getActivationDerivative(INDArray x, Function function){
        switch (function){
            case Tanh:
            	return Nd4j.getExecutioner().execAndReturn(new Tanh(x.dup()).derivative());
            case Logistic:
            	return Nd4j.getExecutioner().execAndReturn(new Sigmoid(x.dup()).derivative());
            case ReLU:
            	return Nd4j.getExecutioner().execAndReturn(new RectifedLinear(x.dup()).derivative());  
            case LeakyReLU:
            	return Nd4j.getExecutioner().execAndReturn(new LeakyReLU(x.dup()).derivative()); 
            case ELU:
            	return Nd4j.getExecutioner().execAndReturn(new ELU(x.dup()).derivative());
            default:
                throw new RuntimeException();
        }
    }
}