package application;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import javax.imageio.ImageIO;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.SnapshotParameters;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.control.ChoiceDialog;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.paint.Color;
import model.DataSetsLoader;
import model.KMeansClassifier;
import model.KnnClassifier;
import model.LabelledDataInstance;

public class MainController {
	//UI ELEMENTS
	@FXML private  Label MainLabel;
	@FXML private  TextArea StatusTA;
	@FXML private  HBox ChartBox;
	@FXML private  TextField KnnTF;
	@FXML private AnchorPane RootAP;
	@FXML private TextField KmeansIterTF;
	private Main main;
	private DataSetsLoader myDataLoader = new DataSetsLoader();
	
	public HashMap<String,XYChart.Series<Number,Number>> chartSeries = new HashMap<>();
	public ScatterChart<Number,Number> scatterChart ;
	public List<String> featuresList = new ArrayList<String>();
	public String xAxisFeature; 
	public String yAxisFeature ;	
	public KMeansClassifier kmeansClassifier;
	public HashMap<String,Integer> axisOptions		= new HashMap<String,Integer>();
	
	
	@SuppressWarnings("static-access")
	public void setMain(Main main) {
		this.main		= main;
		System.out.println(myDataLoader);
		featuresList.add("Petal Width");
		featuresList.add("Petal Length");
		featuresList.add("Sepal Width");
		featuresList.add("Sepal Length");
		axisOptions.put("Petal Length", 0);
		axisOptions.put("Petal Width", 1);
		axisOptions.put("Sepal Length", 2);
		axisOptions.put("Sepal Width", 3);		
		xAxisFeature = "Petal Length";
		yAxisFeature = "Sepal Length";
		final NumberAxis xAxis = new NumberAxis(0, 0, 0.1);
		final NumberAxis yAxis = new NumberAxis(0, 0, 0.1);
		scatterChart			= new ScatterChart<Number,Number>(xAxis,yAxis);

		scatterChart.setAnimated(true);
		scatterChart.getXAxis().setAutoRanging(true);
		scatterChart.getYAxis().setAutoRanging(true);
		scatterChart.setVerticalGridLinesVisible(true);
		ChartBox.setHgrow(scatterChart, Priority.ALWAYS);
		ChartBox.getChildren().add(1, scatterChart);
		LoadIrisDataSet();
		
	}
	

	public void saveCharts() {
		//This Functions will save a chart for the current KNN used in the UI
		// With the X-Axis & Y-Axis Combinations below.
		xAxisFeature 										= "Petal Width";
		yAxisFeature										= "Sepal Width";
		saveChartForFeatureCombination();
		xAxisFeature 										= "Petal Length";
		yAxisFeature										= "Sepal Width";
		saveChartForFeatureCombination();	
		xAxisFeature 										= "Petal Length";
		yAxisFeature										= "Sepal Length";
		saveChartForFeatureCombination();
		xAxisFeature 										= "Petal Width";
		yAxisFeature										= "Sepal Length";
		saveChartForFeatureCombination();
		xAxisFeature 										= "Petal Length";
		yAxisFeature										= "Petal Width";
		saveChartForFeatureCombination();				
		xAxisFeature 										= "Sepal Length";
		yAxisFeature										= "Petal Width";
		saveChartForFeatureCombination();				
		
	}


	private void saveChartForFeatureCombination() {
		scatterChart.getData().clear();
		myDataLoader.clear();		
		LoadIrisDataSet();
		myDataLoader.myclassifier 							= new KnnClassifier(myDataLoader);
		myDataLoader.myclassifier.knnNumber					= Integer.parseInt(KnnTF.getText()); // Get KNN Value from UI	
		computeKnn();
		saveSnapShot();
	}
	
	public void computeKnn() {
		//Clearing the Status TextArea
		StatusTA.setText("");
		scatterChart.getData().clear();
		plotIrisTrainingSetOnChart(xAxisFeature, yAxisFeature);
		// Assigning KNN to Data Set Classifier
		myDataLoader.myclassifier 							= new KnnClassifier(myDataLoader);
		myDataLoader.myclassifier.knnNumber					= Integer.parseInt(KnnTF.getText()); // Get KNN Value from UI
		chartSeries.get("WrongPredictions").getData().clear();

		// Initializing Variables for accuracy calculation Later on
		int count					= 1;
		int correctPrediction		= 0;		
		int trainingInstancesCount	= myDataLoader.trainingDataSetList.size();
		for (LabelledDataInstance testInstance : myDataLoader.testDataSetList) {
			// getting X & Y values based on current Axis Features Selected
			float testXfeature		= testInstance.featureListAsValues.get(featuresList.indexOf(xAxisFeature));
			float testYfeature		= testInstance.featureListAsValues.get(featuresList.indexOf(yAxisFeature));
			
			// Compute the KNN
			String predictedClass	= myDataLoader.myclassifier.predictClassForTestInstance(testInstance);
			testInstance.predictedClass = predictedClass;			
			
			if (predictedClass.equals(testInstance.labelName)) {
				correctPrediction++;
				chartSeries.get(testInstance.labelName).getData().add(new Data<Number,Number>(testXfeature,testYfeature));
				}
			else{
				StatusTA.appendText("FAIL! TestID:"+count+" "+testInstance.labelName
						+"!="+predictedClass+"\n");
				chartSeries.get("WrongPredictions").getData().add(new Data<Number,Number>(testXfeature,testYfeature));
			}
			count++;
		}
		// Calculating Accuracy measure
		myDataLoader.myclassifier.accuracy					= ((double)correctPrediction/(double)trainingInstancesCount)*100.0;				
		StatusTA.appendText("Accuracy @K" + myDataLoader.myclassifier.knnNumber + "NN="+myDataLoader.myclassifier.accuracy) ;
		scatterChart.setTitle(myDataLoader.myclassifier.knnNumber + "NN Accuracy =" + myDataLoader.myclassifier.accuracy);
	}
	
	public void LoadIrisDataSet() {
		System.out.println("Loading IrisDataset");
		myDataLoader.dataSetName = "Iris Dataset";
		myDataLoader.clear();
//						
//		System.out.println(System.getProperty("user.dir").replace('\\', '/') + "/iris-training.txt");
//		System.out.println(getClass().getResource("iris-training.txt").getPath());
		myDataLoader.loadIrisDataSet(System.getProperty("user.dir").replace('\\', '/') + "/iris-training.txt",myDataLoader.trainingDataSetList);		
		myDataLoader.computeRangeForFeaturesInDataSet();
		StatusTA.setText(myDataLoader.toString());
		myDataLoader.loadIrisDataSet(System.getProperty("user.dir").replace('\\', '/') + "/iris-test.txt",myDataLoader.testDataSetList);
		StatusTA.appendText("Test Data Set Size = " + myDataLoader.testDataSetList.size());
		MainLabel.setText("Iris DataSet Loaded");
		scatterChart.getData().clear();
		plotIrisTrainingSetOnChart(xAxisFeature,yAxisFeature);
//		System.out.println(this.MainLabel);
	}
	
	public void updateClustersForNIterations() {
		int iterations					= Integer.parseInt(KmeansIterTF.getText());
		for (int i=0; i<iterations; i++) {
			kmeansClassifier.recomputerCentroids();
			boolean converged			= kmeansClassifier.checkForConvergence();
			if (converged) {
				StatusTA.setText(kmeansClassifier.reportOnClusters()+
						"Clusters Have Converged !!");
				return;
			}
			else {
				kmeansClassifier.updateCentroids();
			}
			kmeansClassifier.clearClusterMap();
			kmeansClassifier.updateCluster();
			StatusTA.setText(kmeansClassifier.reportOnClusters());
			plotClusterPoints();
		}
	}
	public void ClusterByKMeans() {
		myDataLoader.trainingDataSetList.addAll(myDataLoader.testDataSetList);
		myDataLoader.testDataSetList.clear();
		System.out.println(myDataLoader.trainingDataSetList.size());
		scatterChart.getData().clear();
		plotIrisTrainingSetOnChart(xAxisFeature,yAxisFeature);
		StatusTA.setText("Merged Data Sets Total = " + myDataLoader.trainingDataSetList.size()+"\n");
		myDataLoader.clearRanges();
	
		myDataLoader.computeRangeForFeaturesInDataSet();
		kmeansClassifier			= new KMeansClassifier(myDataLoader);
		kmeansClassifier.initialiseRandomClusters();
		System.out.println(kmeansClassifier.clusterPoints);
		kmeansClassifier.updateCluster();
		StatusTA.appendText(myDataLoader.toString());
		StatusTA.appendText(kmeansClassifier.reportOnClusters());
		System.out.println(kmeansClassifier.clusterPoints);
		plotClusterPoints();
		
	}
	private String promptUserForChoice(List<String> dialogData,String message) {		
		ChoiceDialog<String> dialog = new ChoiceDialog<String>(dialogData.get(0), dialogData);
		dialog.setTitle("");
		dialog.setHeaderText(message);		
		Optional<String> result = dialog.showAndWait();
		String selected = "cancelled.";
				
		if (result.isPresent()) {

		    selected = result.get();
		}
		
		return selected;
	}
	public void changeAxisFeature() {
	
		xAxisFeature 							= promptUserForChoice(featuresList,"Please Choose X-Axis Feature");
		yAxisFeature 							= promptUserForChoice(featuresList,"X-Axis = " + xAxisFeature +"\nPlease Choose Y-Axis Feature");
		System.out.println(xAxisFeature + yAxisFeature);
		scatterChart.getData().clear();
		plotIrisTrainingSetOnChart(xAxisFeature, yAxisFeature);
		
	}
	
	
	public void saveAsPng(String fileName) {
		SnapshotParameters snapshotParams   	= new SnapshotParameters();
		snapshotParams.setFill(Color.rgb(40, 40, 40, 1));

		
	    WritableImage image 					= scatterChart.snapshot(snapshotParams,null);
    	
	    
	    File file = new File(fileName+".png");

	    try {
	        ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", file);
	    } catch (IOException e) {
	        
	    }
	}	
	public void saveSnapShot() {
		saveAsPng("Result_" + myDataLoader.myclassifier.knnNumber + "NN_" + xAxisFeature+"_"+yAxisFeature);
	}
	
	
	public void plotClusterPoints() {
		int xAxisFeatureIndex					= axisOptions.get(xAxisFeature);
		int yAxisFeatureIndex					= axisOptions.get(yAxisFeature);
		chartSeries.get("ClusterPoints").getData().clear();
		
		// For QuickNess I have Mapped the 3 Classes Graph Series Randomly To Clusters
		// That way I dont have to add 3 more different colours to the chart 
		HashMap<String,XYChart.Series<Number,Number>> clusterSeries = new HashMap<>();
		clusterSeries.put("Cluster0", chartSeries.get("Iris-versicolor"));
		clusterSeries.put("Cluster1", chartSeries.get("Iris-virginica"));
		clusterSeries.put("Cluster2", chartSeries.get("Iris-setosa"));
		for (String classLabel : myDataLoader.dataSetClasses) {			
			chartSeries.get(classLabel).getData().clear();			
		}
		for (LabelledDataInstance cluster : kmeansClassifier.clusterPoints) {
			float floatx = cluster.featureListAsValues.get(xAxisFeatureIndex);
			float floaty = cluster.featureListAsValues.get(yAxisFeatureIndex);
			for (LabelledDataInstance dataInstance : kmeansClassifier.clusterMap.get(cluster.labelName)) {
				float floatx2 = dataInstance.featureListAsValues.get(xAxisFeatureIndex);
				float floaty2 = dataInstance.featureListAsValues.get(yAxisFeatureIndex);			
				clusterSeries.get(cluster.labelName).getData().add(new Data<Number,Number>(floatx2,floaty2));				
			}
			chartSeries.get("ClusterPoints").getData().add(new Data<Number,Number>(floatx,floaty));
			}			
	}
	public void plotIrisTrainingSetOnChart(String xAxisFeature, String yAxisFeature) {
		// Adding TheLabels of The Data Set as Chart Series Key = String Value = XYChart.Series
		// Later On Referenced for Plotting Correct & Wrong Predictions
		for (String classLabel : myDataLoader.dataSetClasses) {
			XYChart.Series<Number,Number> newSeries	= new XYChart.Series<>();
			newSeries.setName(classLabel);
			chartSeries.put(classLabel, newSeries);
			newSeries.getData().add(new Data<Number,Number>(0,0)); 
			scatterChart.getData().add(newSeries);
			
		}
		
		// Creating a HashMap For FeatureList Manually as Data did not have a header row
		// Later On Used in changeAxisFeature to allow user to plot relationships between different features of the dataSet

		
		int xAxisFeatureIndex					= axisOptions.get(xAxisFeature);
		int yAxisFeatureIndex					= axisOptions.get(yAxisFeature);
		
		
		// 5 attributes - Petal Length , Petal Width , Sepal Length , Sepal width , Class Label
		// ForLoop Below allows me to get the XYChart.Series Obj based on the dataInstance.labelName
		// Plot the dataInstance leaving everything flexible for other data sets in the future
		System.out.println( "Training Data Set "+myDataLoader.trainingDataSetList.size());
		for (LabelledDataInstance dataInstance : myDataLoader.trainingDataSetList) {
			float floatx = dataInstance.featureListAsValues.get(xAxisFeatureIndex);
			float floaty = dataInstance.featureListAsValues.get(yAxisFeatureIndex);			
			chartSeries.get(dataInstance.labelName).getData().add(new Data<Number,Number>(floatx,floaty));
		}
		// Adding Series To The Chart for Wrong Predictions
		XYChart.Series<Number,Number> wrongPredictions	= new XYChart.Series<>();
		wrongPredictions.getData().add(new Data<Number,Number>(0,0));
		wrongPredictions.setName("Wrong Predictions");
		chartSeries.put("WrongPredictions",wrongPredictions);
		scatterChart.getData().add(wrongPredictions);
		
		// Adding Clusters Series To The Chart For KMEANS Representation
		XYChart.Series<Number,Number> clusterPointsSeries	= new XYChart.Series<>();
		clusterPointsSeries.getData().add(new Data<Number,Number>(0,0));
		clusterPointsSeries.setName("ClusterPoints");
		chartSeries.put("ClusterPoints",clusterPointsSeries);
		scatterChart.getData().add(clusterPointsSeries);
		chartSeries.put("ClusterPoints", clusterPointsSeries);
		
		// Setting Axis Labels provided by the function input from changeAxisFeature
		scatterChart.getXAxis().setLabel(xAxisFeature);
		scatterChart.getYAxis().setLabel(yAxisFeature);
		
		
	}
}

