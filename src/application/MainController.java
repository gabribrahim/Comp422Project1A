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
	
	public HashMap<String,Integer> axisOptions		= new HashMap<String,Integer>();
	
	
	@SuppressWarnings("static-access")
	public void setMain(Main main) {
		this.main		= main;
		myDataLoader.loadEmailData(System.getProperty("user.dir").replace('\\', '/') + "/spamLabelled.dat",myDataLoader.trainingDataSetList, true);
//		myDataLoader.loadEmailData(System.getProperty("user.dir").replace('\\', '/') + "/spamUnlabelled.dat",myDataLoader.trainingDataSetList, false);
		
		System.out.println(myDataLoader.trainingDataSetList.size());
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
}

