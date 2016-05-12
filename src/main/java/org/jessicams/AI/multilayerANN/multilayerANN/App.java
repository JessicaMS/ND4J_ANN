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
 * @author Jessica Seibert
 *
 */
public class App 
{

	/**
	 * A helper function to describe a matrix, for debugging
	 * @param description	A string to print out which describes the matrix
	 * @param matrix		An INDArray to display the contents of
	 */
	private static void describeMatrix(String description, INDArray matrix) {
		System.out.println(description);
		System.out.println(""+ matrix.columns() + " x " + matrix.rows() + " Matrix:");
		System.out.println(matrix);
	}

	/**
	 * 
	 * @param dataFilePath
	 * @return
	 */
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

	public static void main( String[] args )  throws Exception
	{
		int numEpochs = 250;
		int inputLayerSize = 2;
		int outputLayerSize = 2;
		int batchSize = 50;
		int outputFreq = 10;

		RecordReader rr = readData("src/main/resources/classification/moon_data_train.csv");
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize, 0, 2);

		INDArray cost;
		NeuralNet myANN = new NeuralNet(trainIter, inputLayerSize, outputLayerSize);
		
		for(int i = 0; i < numEpochs; i++) {
			cost = myANN.batchTrain();
			if (i % outputFreq == 0) {
				System.out.println("Epoch: " + i);
//				System.out.println("Scaled output:");
//				System.out.println(myANN.getYHat().mul(myANN.getOutputScale()));
				System.out.println("Total cost: " + (double)cost.sumNumber().doubleValue() + "\n");
			}
		
		}
		
		
		//Evaluation and testing code copied from DL4J examples
		
		RecordReader rrEval = readData("src/main/resources/classification/moon_data_eval.csv");
		DataSetIterator evalIter = new RecordReaderDataSetIterator(rrEval,batchSize, 0, 2);
		rrEval.reset();
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(outputLayerSize);
        while(evalIter.hasNext()){
            DataSet t = evalIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = myANN.output(features);

            eval.eval(labels, predicted);
        }

        //Print the evaluation statistics
        System.out.println(eval.stats());

        
        
      //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only

        //Plot the data
        double xMin = -1.5;
        double xMax = 2.5;
        double yMin = -1;
        double yMax = 1.5;

        //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
        int nPointsPerAxis = 100;
        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][2];
        int count = 0;
        for( int i=0; i<nPointsPerAxis; i++ ){
            for( int j=0; j<nPointsPerAxis; j++ ){
                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;

                evalPoints[count][0] = x;
                evalPoints[count][1] = y;

                count++;
            }
        }

        INDArray allXYPoints = Nd4j.create(evalPoints);
        INDArray predictionsAtXYPoints = myANN.output(allXYPoints).getColumn(0);

        //Get all of the training data in a single array, and plot it:
        rr.initialize(new FileSplit(new File("src/main/resources/classification/moon_data_train.csv")));
        rr.reset();
        int nTrainPoints = 2000;
        trainIter = new RecordReaderDataSetIterator(rr,nTrainPoints,0,2);
        DataSet ds = trainIter.next();
        PlotUtil.plotTrainingData(ds.getFeatureMatrix(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);


        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        rrEval.initialize(new FileSplit(new File("src/main/resources/classification/moon_data_eval.csv")));
        rrEval.reset();
        int nTestPoints = 1000;
        evalIter = new RecordReaderDataSetIterator(rrEval,nTestPoints,0,2);
        ds = evalIter.next();
        INDArray testPredicted = myANN.output(ds.getFeatureMatrix());
        PlotUtil.plotTestData(ds.getFeatureMatrix(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

        System.out.println("****************Example finished********************");
	}
}
