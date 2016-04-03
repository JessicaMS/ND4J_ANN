package org.jessicams.AI.multilayerANN.multilayerANN;

import java.io.File;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.jessicams.AI.multilayerANN.NeuralNet.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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
			double sum = 0;
			for(int j = 0; j < inputLayer.columns()-1; j++) {
				sum += inputLayer.getDouble(i, j);
			}
			yArray[i] = sum;
		}

		return Nd4j.create(yArray, new int[]{1,inputLayer.rows()});
	}

	private static void describeMatrix(String description, INDArray matrix) {
		System.out.println(description);
		System.out.println(""+ matrix.columns() + " x " + matrix.rows() + " Matrix:");
		System.out.println(matrix);
	}

	public static RecordReader readData() {

		RecordReader rr = new CSVRecordReader();
		try {
			rr.initialize(new FileSplit(new File("src/main/resources/adding.csv")));	
		} catch (Exception e) {
			System.out.println("Read file failed");
			System.out.println(e.getMessage());
		}
				
		return rr;
	}

	public static INDArray inputLayerSingleBatch(RecordReader rr, int batchSize, int labelOffset) {
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,labelOffset,18);
		trainIter.reset();
		System.out.println("HERE");
		
		try {
			DataSet next = trainIter.next();
			
			return next.getFeatureMatrix();
		} catch (Exception e) {
			System.out.println("Next out of bounds");
		}
		
		return null;
	}
	
	public static INDArray outputLayerSingleBatch(RecordReader rr, int batchSize, int labelOffset) {
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,labelOffset-1, 1);
		trainIter.reset();
		System.out.println("HERE");
		
		try {
			DataSet next = trainIter.next();
			
			return next.getLabels();
		} catch (Exception e) {
			System.out.println("Next out of bounds");
		}
		
		return null;
	}

	public static void main( String[] args )  throws Exception
	{
		int numEpochs = 40000;
		int inputLayerSize = 3;
		int trainingSetCount =64;
		int batchSize = 8;
		int outputFreq = 2000;
		//
		INDArray inputLayer;
		INDArray y; 
		
		//inputLayer =  buildInputs(inputLayerSize, trainingSetCount);
		RecordReader rr = readData();
		inputLayer = inputLayerSingleBatch(rr, trainingSetCount, inputLayerSize);
		y = buildOutputs(inputLayer);

		//		
		double inputScale = (Double) inputLayer.maxNumber();
		double outputScale = (Double) y.maxNumber();
		inputLayer = inputLayer.div(inputScale);
		y = y.div(outputScale);   //max in matrix y is scale

		describeMatrix("", inputLayer);
		
		System.out.println(y);
				INDArray yHat, cost;
		
				NeuralNet myANN = new NeuralNet( inputScale, outputScale);
		
				for(int i = 0; i < numEpochs; i++) {
						yHat = myANN.forwardProp(inputLayer);
						cost = myANN.quadraticCost(y.transpose());
						
						
						if (i % outputFreq == 0) {
							System.out.println("Scaled output:");
							System.out.println(yHat.mul(outputScale));
							System.out.println("");
					        System.out.println("Total cost: " + (double)cost.sumNumber().doubleValue());
						}
		
						//Back-propagation
						myANN.costPrime(cost, y, inputLayer);		
		
				}
		
				myANN.testData(new double[]{1.0, 1.0, 1.0});
				
				myANN.testData(new double[]{1.0, 2.0, 1.0});		
				
				myANN.testData(new double[]{3.0, 3.0, 1.0});
				
				myANN.testData(new double[]{8.0, 9.0, 1.0});

	}

	//Plot the data
    static void plot(INDArray x, INDArray y, INDArray... predicted){
        XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");

        for( int i=0; i<predicted.length; i++ ){
            addSeries(dataSet,x,predicted[i],String.valueOf(i));
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Example",      // chart title
                "X",                      // x axis label
                "(X)",         // y axis label
                dataSet,                  // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );

        ChartPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }
    
    private static void addSeries(XYSeriesCollection dataSet, INDArray x, INDArray y, String label){
        double[] xd = x.data().asDouble();
        double[] yd = y.data().asDouble();
        XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }
}
