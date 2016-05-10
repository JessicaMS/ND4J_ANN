package org.jessicams.AI.multilayerANN.multilayerANN;

import java.io.File;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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


	private static void describeMatrix(String description, INDArray matrix) {
		System.out.println(description);
		System.out.println(""+ matrix.columns() + " x " + matrix.rows() + " Matrix:");
		System.out.println(matrix);
	}

	public static RecordReader readData(String dataFilePath) {
		RecordReader rr = new CSVRecordReader();
		try {
			rr.initialize(new FileSplit(new File(dataFilePath)));	
		} catch (Exception e) {
			System.out.println("Read file failed");
			System.out.println(e.getMessage());
		}
		return rr;
	}

	public static INDArray inputLayerSingleBatch(RecordReader rr, int batchSize, int labelOffset) {
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize);
		trainIter.reset();
		try {
			DataSet next = trainIter.next();
			return next.getFeatureMatrix();
		} catch (Exception e) {
			System.out.println("Next out of bounds");
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
		return null;
	}
	

	public static void main( String[] args )  throws Exception
	{
		int numEpochs = 500;
		int inputLayerSize = 2;
		int outputLayerSize = 2;
		int batchSize = 25;
		int outputFreq = 10;

		RecordReader rr = readData("src/main/resources/classification/moon_data_train.csv");
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize, 0, 2);
		rr = readData("src/main/resources/classification/moon_data_eval.csv");
		DataSetIterator evalIter = new RecordReaderDataSetIterator(rr,batchSize, 0, 2);
		//inputLayer = inputLayerSingleBatch(rr, trainingSetCount, inputLayerSize);

		INDArray cost;
		NeuralNet myANN = new NeuralNet(trainIter, inputLayerSize, outputLayerSize);
		
		for(int i = 0; i < numEpochs; i++) {
			cost = myANN.batchTrain();
			if (i % outputFreq == 0) {
				System.out.println("Epoch: " + i);
				System.out.println("Scaled output:");
				System.out.println(myANN.getYHat().mul(myANN.getOutputScale()));
				System.out.println("Total cost: " + (double)cost.sumNumber().doubleValue() + "\n");
			}
		
		}
		//plot(inputLayer,myANN.x,getFunctionValues(x,fn),networkPredictions);
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(outputLayerSize);
        while(evalIter.hasNext()){
            DataSet t = evalIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = myANN.forwardProp(features);

            eval.eval(labels, predicted);
        }

        //Print the evaluation statistics
        System.out.println(eval.stats());

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
