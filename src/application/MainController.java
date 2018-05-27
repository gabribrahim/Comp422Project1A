package application;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;

import javax.imageio.ImageIO;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.SnapshotParameters;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.control.ChoiceDialog;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.paint.Color;
import model.DataSetsLoader;


import javax.script.ScriptEngineManager;
import javax.script.ScriptEngine;
import javax.script.ScriptException;

import model.LabelledDataInstance;

public class MainController {
	//UI ELEMENTS
	@FXML private  Label MainLabel;
	@FXML private  TextArea StatusTA;
	@FXML private  HBox ChartBox;
	@FXML private  TextField KnnTF;
	@FXML private AnchorPane RootAP;
	@FXML private TextField KmeansIterTF;
    private TableView countTable 			= new TableView();
    private TableView probabiltyTable 		= new TableView();
    TableColumn classCol 					= new TableColumn("Class");
    TableColumn probCol 					= new TableColumn("Probability");
    TableColumn noSpamCol 					= new TableColumn("0 [Not Spam]");
    TableColumn noSpamCol2 					= new TableColumn("0 [Not Spam]");
    TableColumn spamCol 					= new TableColumn("1 [Spam]");
    TableColumn spamCol2 					= new TableColumn("1 [Spam]");
    
	private Main main;
	private DataSetsLoader myDataLoader 	= new DataSetsLoader();
	
	public HashMap<String,XYChart.Series<Number,Number>> chartSeries = new HashMap<>();
	public ScatterChart<Number,Number> scatterChart ;
	public List<String> featuresList = new ArrayList<String>();
	public String xAxisFeature; 
	public String yAxisFeature ;	
	
	public HashMap<String,Integer> axisOptions		= new HashMap<String,Integer>();
	
	
	@SuppressWarnings("static-access")
	public void setMain(Main main) {
		//G7244
		this.main		= main;

//		myDataLoader.loadEmailData(System.getProperty("user.dir").replace('\\', '/') + "/spamUnlabelled.dat",myDataLoader.trainingDataSetList, false);
		setupTables();
		loadLabelledData();
		createCountTable();
		createProbTable();
		loadUnLabelledData();
		writeOutputFile();
		
	}
	public void writeOutputFile() {
		ArrayList<String> linesToWrite	= new ArrayList<>();
		for (LabelledDataInstance instance : myDataLoader.testDataSetList) {
//			System.out.println(instance.featureListAsValues+">>"+instance.featureListAsValues.size());
			String formula				= "Instance [ "+myDataLoader.testDataSetList.indexOf(instance)+" ]\n"+
						instance.featureListAsValues+"\nP(S|D)  = ";
			formula 					= assembleFormulaSymbolic(instance, formula);
			formula 					+="\n\n";
			formula 					= assembleFormulaNumerical(instance, formula);
			try {
				formula						= evaluateFormulaNumerical(instance, formula);
			} catch (ScriptException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			formula +="\n========================================================================";
			linesToWrite.add(formula);
			System.out.println(formula);
		}
		Path file = Paths.get("output.txt");
		try {
			Files.write(file, linesToWrite, Charset.forName("UTF-8"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}			
	}
	private String assembleFormulaSymbolic(LabelledDataInstance instance, String formula) {
		for (int i =0 ; i<instance.featureListAsValues.size();i++) {
			boolean featureState	= instance.featureListAsValues.get(i);
			if (featureState) {
				formula				+= "P(Feature"+i+"=true|S).";
			}else {
				formula				+= "P(Feature"+i+"=false|S).";
			}
		}
		formula						+= "P(S)\nP(S`|D) = ";
		for (int i =0 ; i<instance.featureListAsValues.size();i++) {
			boolean featureState	= instance.featureListAsValues.get(i);
			if (featureState) {
				formula				+= "P(Feature"+i+"=true|S`).";
			}else {
				formula				+= "P(Feature"+i+"=false|S`).";
			}
		}	
		formula						+= "P(S`)";
		return formula;
	}
	
	private String assembleFormulaNumerical(LabelledDataInstance instance, String formula) {
		formula						+= "P(S|D)  = ";
		for (int i =0 ; i<instance.featureListAsValues.size();i++) {
			boolean featureState	= instance.featureListAsValues.get(i);
			if (featureState) {
				formula				+= myDataLoader.probTable.get("Feature"+i+"=true").get("1")+" X ";
			}else {
				formula				+= myDataLoader.probTable.get("Feature"+i+"=false").get("1")+" X ";
			}
		}
		formula						+= myDataLoader.probTable.get("P(Class)").get("1")+"\nP(S`|D) = ";
		for (int i =0 ; i<instance.featureListAsValues.size();i++) {
			boolean featureState	= instance.featureListAsValues.get(i);
			if (featureState) {
				formula				+= myDataLoader.probTable.get("Feature"+i+"=true").get("0")+" X ";
			}else {
				formula				+= myDataLoader.probTable.get("Feature"+i+"=false").get("0")+" X ";
			}
		}	
		formula						+=  myDataLoader.probTable.get("P(Class)").get("0");
		return formula;
	}	
	
	private String evaluateFormulaNumerical(LabelledDataInstance instance, String formula) throws ScriptException {		
		Float spamProb				=(float) 1;
		Float noSpamProb			=(float) 1;	
	    ScriptEngineManager mgr 	= new ScriptEngineManager();
	    ScriptEngine engine 		= mgr.getEngineByName("JavaScript");		
		for (int i =0 ; i<instance.featureListAsValues.size();i++) {
			boolean featureState	= instance.featureListAsValues.get(i);
			if (featureState) {
				String toEval		= myDataLoader.probTable.get("Feature"+i+"=true").get("1");
				String evaluatedFloat= engine.eval(toEval).toString();
				spamProb			*= Float.parseFloat(evaluatedFloat);
						
				 
			}else {
				String toEval		= myDataLoader.probTable.get("Feature"+i+"=false").get("1");
				String evaluatedFloat= engine.eval(toEval).toString();
				spamProb			*= Float.parseFloat(evaluatedFloat);			}
		}
		String toEvalClassProb		= myDataLoader.probTable.get("P(Class)").get("1");
		String evaluatedClassProb	= engine.eval(toEvalClassProb).toString();
		spamProb					*= Float.parseFloat(evaluatedClassProb);		
//		formula						+= myDataLoader.probTable.get("P(Class)").get("1")+"\nP(S`|D) = ";
		
		for (int i =0 ; i<instance.featureListAsValues.size();i++) {
			boolean featureState	= instance.featureListAsValues.get(i);
			if (featureState) {
				String toEval		= myDataLoader.probTable.get("Feature"+i+"=true").get("0");
				String evaluatedFloat= engine.eval(toEval).toString();
				noSpamProb			*= Float.parseFloat(evaluatedFloat);
			}else {
				String toEval		= myDataLoader.probTable.get("Feature"+i+"=false").get("0");
				String evaluatedFloat= engine.eval(toEval).toString();
				noSpamProb			*= Float.parseFloat(evaluatedFloat);			
				}
		}	
//		formula						+=  myDataLoader.probTable.get("P(Class)").get("0");
		toEvalClassProb				= myDataLoader.probTable.get("P(Class)").get("0");
		evaluatedClassProb			= engine.eval(toEvalClassProb).toString();
		noSpamProb					*= Float.parseFloat(evaluatedClassProb);		
		NumberFormat floatFormatter	= NumberFormat.getInstance();
		floatFormatter.setMaximumFractionDigits(14);
		floatFormatter.setGroupingUsed(false);
		formula						+= "\nP(S|D)  = " + floatFormatter.format(spamProb)+"\nP(S`|D) = "+floatFormatter.format(noSpamProb);
		if(spamProb>noSpamProb) {
			formula					+= "\nClass   = SPAM\nSince Prob of Spam > Prob of No Spam, then email is classified as SPAM\n";
		}else {
			formula					+= "\nClass   = NOTSPAM\nSince Prob of Spam < Prob of No Spam, then email is classified as NOTSPAM";
		}
		return formula;
	}	
	
	
	public void loadUnLabelledData() {
		myDataLoader.loadEmailData(System.getProperty("user.dir").replace('\\', '/') + "/spamUnlabelled.dat",myDataLoader.testDataSetList, false);
		updateStatusText("Loaded Unlabeled Data Set\n"+ myDataLoader.testDataSetList.size()+" Instances");
	}	
	public void loadLabelledData() {
		myDataLoader.loadEmailData(System.getProperty("user.dir").replace('\\', '/') + "/spamLabelled.dat",myDataLoader.trainingDataSetList, true);
		updateStatusText("Loaded Labeled Data Set\n"+ myDataLoader.trainingDataSetList.size()+" Instances");
	}
	public void createProbTable() {
		probabiltyTable.getItems().clear();
		myDataLoader.createProbTable();
		for (Entry<String, HashMap<String, String>> entry:myDataLoader.probTable.entrySet()) {			
			TableModelRepUI uiEntry		= new TableModelRepUI(entry.getKey(),entry.getValue().get("1"),entry.getValue().get("0"),"");
			probabiltyTable.getItems().add(uiEntry);
			
		}
		
	}
	public void createCountTable() {
		myDataLoader.createCountTable();
		countTable.getItems().clear();
		System.out.println(myDataLoader.classesCount);
		updateStatusText("DataSet Count = "+ myDataLoader.trainingDataSetList.size()+
				"\nClasses Count = "+myDataLoader.classesCount);			
//		TableModelRepUI totalsEntry		= new TableModelRepUI("Total", myDataLoader.countTable.get("Total").get("1").toString(), 
//				myDataLoader.countTable.get("Total").get("0").toString(), "");
//		countTable.getItems().add(totalsEntry);	
		for (Entry<String, HashMap<String, Integer>> entry:myDataLoader.countTable.entrySet()) {
			TableModelRepUI uiEntry		= new TableModelRepUI(entry.getKey(),entry.getValue().get("1").toString(),
					entry.getValue().get("0").toString(),"");
			countTable.getItems().add(uiEntry);
			
		}
	}
	public void setupTables() {
        countTable.setEditable(true);
        probabiltyTable.setEditable(true);

        ChartBox.setHgrow(countTable, Priority.ALWAYS);
        ChartBox.setHgrow(probabiltyTable, Priority.ALWAYS);
        countTable.getColumns().addAll(classCol, noSpamCol, spamCol);
        probabiltyTable.getColumns().addAll(probCol, noSpamCol2, spamCol2);
		ChartBox.getChildren().add(countTable);
		ChartBox.getChildren().add(probabiltyTable);
//		TableModelRepUI test = new TableModelRepUI("TESTAAaaaaaaaaaaaaaa", "aa", "aaa", "vvvv");
//		probabiltyTable.getItems().add(test);
		spamCol.setCellValueFactory(new PropertyValueFactory<TableModelRepUI,String>("spam"));
		classCol.setCellValueFactory(new PropertyValueFactory<TableModelRepUI,String>("className"));
		noSpamCol.setCellValueFactory(new PropertyValueFactory<TableModelRepUI,String>("NoSPam"));
		
		spamCol2.setCellValueFactory(new PropertyValueFactory<TableModelRepUI,String>("spam"));
		probCol.setCellValueFactory(new PropertyValueFactory<TableModelRepUI,String>("className"));
		noSpamCol2.setCellValueFactory(new PropertyValueFactory<TableModelRepUI,String>("NoSPam"));
		
	}
	public void updateStatusText(String message) {
		StatusTA.setText(message+"\n");
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

