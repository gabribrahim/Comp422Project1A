package application;

import java.awt.FileDialog;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.geometry.Bounds;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.SnapshotParameters;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.control.Alert;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Labeled;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TextInputDialog;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.paint.Color;
import model.DataSetsLoader;
import model.ImageConvolutionMask;

import javax.script.ScriptEngineManager;
import javax.script.ScriptEngine;
import javax.script.ScriptException;
import javax.swing.JFileChooser;

import model.LabelledDataInstance;

import org.omg.CORBA.FieldNameHelper;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import com.sun.deploy.uitoolkit.impl.fx.ui.FXConsole;

import java.lang.reflect.Field;
import java.net.NetworkInterface;

import org.opencv.highgui.HighGui;


import javafx.collections.FXCollections;
public class MainController {
	//UI ELEMENTS
	@FXML private  Label MainLabel;
	@FXML private  TextArea StatusTA;
	@FXML private  HBox ChartBox;
	@FXML private  TextField KnnTF;
	@FXML private AnchorPane RootAP;	
	@FXML private Group SourceImgGRP;
	@FXML private ScrollPane ImgScrollPane;
	@FXML private GridPane ConvMaskGridP; 
	@FXML private HBox ConvBox;
	@FXML private CheckBox ToBufferImage;
	@FXML private ComboBox<String> FeaturesDB;
    @FXML private CheckBox ShowFeatureRoiCB;
    @FXML private CheckBox ApplyFaceFeatureCB;
    
	private Main main;
	private DataSetsLoader myDataLoader 							= new DataSetsLoader();
	public int effectIterations										= 0;
	public List<String> featuresList 								= new ArrayList<String>();
	public Mat loadedImage;
	public String path 												= null;
	public ImageView imgView;
	public Float currentZoomFactor 									= 1.0f;
	public HashMap<ArrayList<Integer>,Float> convolutionKernel		= new HashMap<ArrayList<Integer>,Float>();
	public Mat imageBufA;
	public Mat imageBufB;
	public Mat bufferImage;
	public Alert alert = new Alert(Alert.AlertType.INFORMATION);
	Map<String, Runnable> features 									= new HashMap<>();
	LinkedHashMap<String,String> imageFeaturesVector				= new LinkedHashMap<>();
	ArrayList<HashMap<String,String>> trainingFaceSet				= new ArrayList<>();
	ArrayList<HashMap<String,String>> trainingNoneFaceSet			= new ArrayList<>();
	ArrayList<HashMap<String,String>> testFaceSet					= new ArrayList<>();
	ArrayList<HashMap<String,String>> testNoneFaceSet				= new ArrayList<>();
	HashMap<String,ArrayList<Integer>> featuresRoIs					= new HashMap<>();
	
	@SuppressWarnings("static-access")
	public void setMain(Main main) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//		FeaturesDP.getChildrenUnmodifiable().add();
		this.main		= main;
		
		aSetupForFeatures();
//		loadEdgeDetectionImage();
//		test();
//		applyConvolutionOperator();
//		System.exit(0);
		loadRandomFaceImage();
//		features.get("FeatureA").run();
//		System.exit(0);
//		createVectorFeatures();
	}

	private void aSetupForFeatures() {
		//All Features UI ADD Setup
		FeaturesDB.getItems().addAll("noseBridge","forehead","eyeLeft","eyeRight","cheekBonesLeft","cheekBonesRight","symmetryX","mouth");
		FeaturesDB.setValue("noseBridge");
		
		//Forehead
		
		features.put("forehead", () -> foreheadPrepProcess());	
		featuresRoIs.put("forehead", new ArrayList<Integer>() {{add(0);add(0);add(18);add(2);}});
		
		// Mouth
		features.put("mouth", () -> mouthBridge());	
		featuresRoIs.put("mouth", new ArrayList<Integer>() {{add(1);add(12);add(16);add(5);}});
		
		// SymmetryX
		features.put("symmetryX", () -> checkSymmetryXScore());	
		featuresRoIs.put("symmetryX", new ArrayList<Integer>() {{add(0);add(0);add(9);add(18);}});
		
		//Nose Bridge Feature Setup ROI = {x,y,width,height}
		features.put("noseBridge", () -> noseBridge());	
		featuresRoIs.put("noseBridge", new ArrayList<Integer>() {{add(7);add(0);add(4);add(9);}});
		
		//Eyes Feature Setup ROI = {x,y,width,height}
		
		features.put("eyeRight", () -> eyesPreprocess());	
		features.put("eyeLeft", () -> eyesPreprocess());
		featuresRoIs.put("eyeRight", new ArrayList<Integer>() {{add(11);add(1);add(7);add(5);}});
		featuresRoIs.put("eyeLeft", new ArrayList<Integer>() {{add(0);add(1);add(7);add(5);}});
		
		//Cheek Bones Feature Setup ROI = {x,y,width,height}		
		features.put("cheekBonesLeft", () -> cheekBonesPrepProcess());
		features.put("cheekBonesRight", () -> cheekBonesPrepProcess());	
		featuresRoIs.put("cheekBonesLeft", new ArrayList<Integer>() {{add(0);add(0);add(6);add(10);}});
		featuresRoIs.put("cheekBonesRight", new ArrayList<Integer>() {{add(12);add(0);add(6);add(10);}});
	}
	
	@SuppressWarnings("unchecked")
	public void getConvolutionKernelFromUI() {
		convolutionKernel.clear();
		ScriptEngineManager manager = new ScriptEngineManager();
		ScriptEngine engine 		= manager.getEngineByName("js");
			
		for ( Node valueUI :ConvMaskGridP.getChildren()) {			
			if (valueUI.idProperty().getValue() != null) {
//				System.out.print(valueUI.idProperty().getValue()+" ");
				String id 			= valueUI.idProperty().getValue();
				ArrayList<Integer> pixelpos = new ArrayList<Integer>();
				try {
					Object result 	= engine.eval(((TextField) valueUI).getText());
					switch (id) {
					case "Conv00TF":
						pixelpos.add(-1);
						pixelpos.add(-1);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();
					case "Conv01TF":
						pixelpos.add(-1);
						pixelpos.add(-0);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();
					case "Conv02TF":
						pixelpos.add(-1);
						pixelpos.add(1);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();	
					case "Conv10TF":
						pixelpos.add(0);
						pixelpos.add(-1);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();
					case "Conv11TF":
						pixelpos.add(0);
						pixelpos.add(0);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();	
					case "Conv12TF":
						pixelpos.add(0);
						pixelpos.add(1);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();
					case "Conv20TF":
						pixelpos.add(1);
						pixelpos.add(-1);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();
					case "Conv21TF":
						pixelpos.add(1);
						pixelpos.add(0);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();							
					case "Conv22TF":
						pixelpos.add(1);
						pixelpos.add(1);
						convolutionKernel.put((ArrayList<Integer>) pixelpos.clone(), Float.parseFloat(result.toString()));
						pixelpos.clear();							
					}
//					System.out.println(Float.parseFloat(result.toString()));
				} catch (ScriptException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}	
			}
		}
//		convolutionKernel.put(key, value);
	}
	
	public void threshold() {
		ToBufferImage.setSelected(true);
		meanFilter();
		int threshold					= getUserInt("40", "WhiteThreshold", "Levels of White To Keep", "intensity<x then = 0");
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 6);
		double newPixel;
		for (int col=0; col<loadedImage.size().width;col++) {
			for (int row=0; row<loadedImage.size().height;row++) {
				newPixel				= loadedImage.get(row, col)[0];
				if(Math.round(newPixel)<threshold) {
					newPixel			= 0;
				}else {
					newPixel			=255;
				}
				double[] p				= {newPixel,newPixel,newPixel};
				bufferImage.put(row, col, p);
			}			
		}
		if (ToBufferImage.isSelected()) {
			loadedImage					= bufferImage.clone();
		}		
		showLoadedImageInUi(bufferImage);		
		
	}
	public void medianFilter() {
		int maskSize					= getUserInt("3", "Kernel Size", "3x3 or 5x5", "Please provide ONE Integer");
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, maskSize);
		mask.calculateRelPixelPositionsInMask();		
		mask.loadedImage				= loadedImage;
		mask.convolutionKernel			= convolutionKernel;
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 6);
		double newPixel;
		for (int col=0; col<loadedImage.size().width;col++) {
			for (int row=0; row<loadedImage.size().height;row++) {
				mask.pixelX =col;
				mask.pixelY = row;
				mask.calculateAbsPixelPositionsInMask();
//				System.out.println(mask.pixelsUnderMask);
				newPixel				= mask.calculateMedianFilterValue();
				double[] p				= {newPixel,newPixel,newPixel};
				bufferImage.put(row, col, p);
			}			
		}
		if (ToBufferImage.isSelected()) {
			loadedImage					= bufferImage.clone();
		}		
		showLoadedImageInUi(bufferImage);		
	}
	
	public void meanFilter() {
		int maskSize					= getUserInt("3", "Kernel Size", "3x3 or 5x5", "Please provide ONE Integer");
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, maskSize);
		mask.calculateRelPixelPositionsInMask();		
		mask.loadedImage				= loadedImage;
		mask.convolutionKernel			= convolutionKernel;
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 6);
		double newPixel;
		for (int col=0; col<loadedImage.size().width;col++) {
			for (int row=0; row<loadedImage.size().height;row++) {
				mask.pixelX =col;
				mask.pixelY = row;
				mask.calculateAbsPixelPositionsInMask();
				newPixel				= mask.calculateMeanFilterValue();
				double[] p				= {newPixel,newPixel,newPixel};
				bufferImage.put(row, col, p);
			}
		}
		if (ToBufferImage.isSelected()) {
			loadedImage					= bufferImage.clone();
		}		
		showLoadedImageInUi(bufferImage);		
	}
	
	public void showMessage(String message) {
		alert.setHeaderText(message);
		alert.show();
		
	}
	public void createVectorFeatures() {
		Instant start = Instant.now();
		
//		// Train Faces
		String directory				=System.getProperty("user.dir").replace('\\', '/') + "/mit-cbcl-faces-balanced/train/face";
		File aDirectory 				= new File(directory);
		String[] filesInDir 			= aDirectory.list();
		vectorizeImagesInDirectory(directory, filesInDir,"trainingFace");
		
		// Train None Faces
		directory						= System.getProperty("user.dir").replace('\\', '/') + "/mit-cbcl-faces-balanced/train/non-face";
		aDirectory 						= new File(directory);
		filesInDir 						= aDirectory.list();
		vectorizeImagesInDirectory(directory, filesInDir,"trainNoneFace");

		// Test Faces
		directory						= System.getProperty("user.dir").replace('\\', '/') + "/mit-cbcl-faces-balanced/test/face";
		aDirectory 						= new File(directory);
		filesInDir 						= aDirectory.list();
		vectorizeImagesInDirectory(directory, filesInDir,"testFace");

		// Test None Faces
		directory						= System.getProperty("user.dir").replace('\\', '/') + "/mit-cbcl-faces-balanced/test/non-face";
		aDirectory 						= new File(directory);
		filesInDir 						= aDirectory.list();
		vectorizeImagesInDirectory(directory, filesInDir,"testNoneFace");

		writeCsvFiles();
		Instant finish = Instant.now();
		long timeElapsed = Duration.between(start, finish).toMillis()/1000;
		System.out.println("TimeForVectorizeImage = "+timeElapsed);
	}
	
	public void writeCsvFiles() {
		ArrayList<String>columnLabels 				= new ArrayList<>();
		for (String label:imageFeaturesVector.keySet())columnLabels.add(label);
		
		HashMap<String,ArrayList<HashMap<String,String>>> entriesToWrite = new HashMap<>();
		entriesToWrite.put("trainingFace", trainingFaceSet);
		entriesToWrite.put("testFace", testFaceSet);
		entriesToWrite.put("testNoneFace", testNoneFaceSet);
		entriesToWrite.put("trainNoneFace", trainingNoneFaceSet);
		
		
		for ( String entryToWrite:entriesToWrite.keySet()) {
			ArrayList<String> linesToWrite 		= new ArrayList<>();
			String columnLabelsLine 			= String.join(",", columnLabels);
			linesToWrite.add(columnLabelsLine);
			ArrayList<HashMap<String,String>> imageEntries = entriesToWrite.get(entryToWrite);			
			for(HashMap<String,String> imageFeaturesVector:imageEntries) {
//				System.out.println(imageFeaturesVector.values());
				ArrayList<String> values		= new ArrayList<>();
				for (String feature:columnLabels) {
					values.add(imageFeaturesVector.get(feature));
				}
				String featuresLine 			= String.join(",", values);
				linesToWrite.add(featuresLine);
			}
			Path file = Paths.get(entryToWrite+".csv");
			try {
				Files.write(file, linesToWrite, Charset.forName("UTF-8"));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				showMessage("Failed To Write New File :-(\nReasons Could be:.\n"+
				"1.File is Opened By Another Software\n"+
				"2.User Does not Have Permissions To Write to Folder\n"+
				"Please kindly check the above issues & Try Again.");
				e.printStackTrace();
				alert.showAndWait();
			}	
			
		}
		trainingFaceSet.clear();
		trainingNoneFaceSet.clear();
		testFaceSet.clear();
		testNoneFaceSet.clear();
		
	}

	public void vectorizeImagesInDirectory(String directory, String[] filesInDir,String dataSet) {
//		trainingFaceSet.clear();
//		trainingNoneFaceSet.clear();
//		testFaceSet.clear();
//		testNoneFaceSet.clear();
		for (int index=0;index<filesInDir.length;index++) {
			path							= directory +"/"+filesInDir[index];
//			System.out.println(path);
			loadedImage						= Imgcodecs.imread(path);		
			createVectorFeature();
			alert.hide();	
			if (dataSet=="trainingFace") {
				imageFeaturesVector.put("label","1");
				trainingFaceSet.add((HashMap<String, String>) imageFeaturesVector.clone());
			}
			if (dataSet=="testFace") {
				imageFeaturesVector.put("label","1");
				testFaceSet.add((HashMap<String, String>) imageFeaturesVector.clone());
			}
			if (dataSet=="testNoneFace") {
				imageFeaturesVector.put("label","0");
				testNoneFaceSet.add((HashMap<String, String>) imageFeaturesVector.clone());
			}
			if (dataSet=="trainNoneFace") {
				imageFeaturesVector.put("label","0");
				trainingNoneFaceSet.add((HashMap<String, String>) imageFeaturesVector.clone());
			}
//			System.out.println(imageFeaturesVector);
			showMessage(dataSet + " DataSet\nVectorized Image "+ index +"/"+filesInDir.length);
//			if (index==10)break;
		}
	}
	
	public void computeStatsForRect(String roiName, String featureLabel) {
		Double average					= averageScoreForROI(roiName);
		Double std						= stdScoreForROI(roiName,average);
		Double momentum					= momentumScoreForROI(roiName);
		imageFeaturesVector.put(featureLabel+"Average", average.toString());
		imageFeaturesVector.put(featureLabel+"Std", std.toString());
		imageFeaturesVector.put(featureLabel+"Momentum", momentum.toString());
		
	}
	public void createVectorFeature() {
		imageFeaturesVector.clear();
		File imagePath					= new File(path);
		imageFeaturesVector.put("filePath",imagePath.getAbsolutePath());
		
		reloadImageFromDisk();
		Double symmetryX				= checkSymmetryXScore();
		imageFeaturesVector.put("symmetryX", symmetryX.toString());
		ToBufferImage.setSelected(true);
		
		// Features with Raw Image//
		computeStatsForRect("noseBridge", "noseUnfiltered");
		computeStatsForRect("cheekBonesLeft", "cheekBonesLeftUnfiltered");
		computeStatsForRect("cheekBonesRight", "cheekBonesRightUnfiltered");
		computeStatsForRect("eyeLeft", "eyeLeftUnfiltered");
		computeStatsForRect("eyeRight", "eyeRightUnfiltered");
		computeStatsForRect("mouth", "mouthUnfiltered");
		computeStatsForRect("forehead", "foreheadUnfiltered");
		
		// Features With Pre-processed Masks//
		//Nose Bridge//
		features.get("noseBridge").run();
		computeStatsForRect("noseBridge", "nose");

		//CheekBones//
		reloadImageFromDisk();
		features.get("cheekBonesLeft").run();
		computeStatsForRect("cheekBonesLeft", "cheekBonesLeft");
		computeStatsForRect("cheekBonesRight", "cheekBonesRight");
		
		//Eyes//
		reloadImageFromDisk();
		features.get("eyeLeft").run();
		ArrayList<Double> eyeAverages	= eyesAverageScore();
		imageFeaturesVector.put("leftEyeAverage", eyeAverages.get(0).toString());
		imageFeaturesVector.put("rightEyeAverage", eyeAverages.get(1).toString());
		// I DONT BELIEVE STD IS RELEVANT FOR MASKED EYES THEY ARE ALWAYS SPARSE PIXELS
		
		//Mouth//
		reloadImageFromDisk();
		features.get("mouth").run();
		computeStatsForRect("mouth", "mouth");
		
		//Forehead//
		reloadImageFromDisk();
		features.get("forehead").run();
		computeStatsForRect("forehead", "forehead");
		String message = "";
		for (String featureName :imageFeaturesVector.keySet()) {
			if (featureName.equals("filePath"))continue;
			message+=featureName+"="+Math.round(Double.parseDouble(imageFeaturesVector.get(featureName)))+"\n";
		}
		updateStatusText(message);
//		updateStatusText("Nose Bridge = " + Math.round(noseBridgeAverage)+
//						"\nCheekBonesLeft = "+Math.round(leftCheekAverage)
//						+"\nCheekBonesRight = "+Math.round(rightCheekAverage)+
//						"\nEyeLeft = "+Math.round(eyeAverages.get(0))
//						+"\nEyeRight = "+Math.round(eyeAverages.get(1))
//						+"\nMouth = "+Math.round(mouthAverage)
//						+"\nForeHead = "+Math.round(foreheadAverage)
//						+"\nSymmetry = "+symmetryX);
		
	}
	
	
	
	public ArrayList<Double> eyesAverageScore() {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		
		mask.loadedImage				= loadedImage;
		ArrayList<Integer> roi1			= featuresRoIs.get("eyeRight");
		ArrayList<Integer> roi2			= featuresRoIs.get("eyeLeft");
		
		LinkedHashMap<ArrayList<Integer>,Double> pixelsInRect = mask.getRect(roi1.get(0),roi1.get(1),roi1.get(2),roi1.get(3));
		boolean isRightEyeConstant		= mask.checkifRectIsConstantIntensity(pixelsInRect);
		double rightIntensity			= mask.computeAverageForRect(pixelsInRect);
		pixelsInRect					= mask.getRect(roi2.get(0),roi2.get(1),roi2.get(2),roi2.get(3));
		boolean isLeftEyeConstant		= mask.checkifRectIsConstantIntensity(pixelsInRect);
		double leftIntensity			= mask.computeAverageForRect(pixelsInRect);
		
		if (isLeftEyeConstant | isRightEyeConstant) {
			return new ArrayList<Double>() {{add(0.0);add(0.0);}};
		}
		
		return new ArrayList<Double>() {{add(leftIntensity);add(rightIntensity);}};
	}	
	
	public ArrayList<Double> cheekBonesAverageScore() {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.loadedImage				= loadedImage;
		ArrayList<Integer> roi1			= featuresRoIs.get("cheekBonesRight");
		ArrayList<Integer> roi2			= featuresRoIs.get("cheekBonesLeft");
		
		LinkedHashMap<ArrayList<Integer>,Double> pixelsInRect = mask.getRect(roi1.get(0),roi1.get(1),roi1.get(2),roi1.get(3));		
		double rightIntensity			= mask.computeAverageForRectWithThreshold(pixelsInRect,0,255);
		pixelsInRect					= mask.getRect(roi2.get(0),roi2.get(1),roi2.get(2),roi2.get(3));
		double leftIntensity			= mask.computeAverageForRectWithThreshold(pixelsInRect,0,255);
		
		
		return new ArrayList<Double>() {{add(leftIntensity);add(rightIntensity);}};
	}

	public double stdScoreForROI(String roiName,double average) {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.loadedImage				= loadedImage;
		ArrayList<Integer> roi			= featuresRoIs.get(roiName);
		LinkedHashMap<ArrayList<Integer>,Double> pixelsInRect = mask.getRect(roi.get(0),roi.get(1),roi.get(2),roi.get(3));		
		double totalIntensity			= mask.computeDeviationForRect(pixelsInRect, average);	
		return totalIntensity;
	}	
	
	public double momentumScoreForROI(String roiName) {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.loadedImage				= loadedImage;
		ArrayList<Integer> roi			= featuresRoIs.get(roiName);
		LinkedHashMap<ArrayList<Integer>,Double> pixelsInRect = mask.getRect(roi.get(0),roi.get(1),roi.get(2),roi.get(3));		
		double totalIntensity			= mask.compute1MomentumForRect(pixelsInRect);	
		return totalIntensity;
	}
	
	public double averageScoreForROI(String roiName) {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.loadedImage				= loadedImage;
		ArrayList<Integer> roi			= featuresRoIs.get(roiName);
		LinkedHashMap<ArrayList<Integer>,Double> pixelsInRect = mask.getRect(roi.get(0),roi.get(1),roi.get(2),roi.get(3));		
		double totalIntensity			= mask.computeAverageForRect(pixelsInRect);	
		return totalIntensity;
	}	
	
	
	public double checkSymmetryXScore() {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.loadedImage				= loadedImage;
		updateStatusText("SymmetryX score " +mask.calculateSymmetryX());
		return mask.calculateSymmetryX();
	}
	public void test5() {
		
		path							= System.getProperty("user.dir").replace('\\', '/') + "/face00008.pgm";
//		path							= System.getProperty("user.dir").replace('\\', '/') + "/test-pattern.tif";
		loadedImage						= Imgcodecs.imread(path);
		currentZoomFactor				= 10.0f;
		System.out.println(loadedImage);
			
		

//		mask.convolutionKernel			= convolutionKernel;
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 0);
//		highlightRect(6,2,6,9);

	}

	public void highlightRect(int x, int y, int width, int height) {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
//		System.out.println("IMAGE DEPTH = "+ loadedImage.depth() + " IMAGE TYPE = "+loadedImage.type());
//		Imgproc.cvtColor(loadedImage.clone(), loadedImage, 0);
		if (loadedImage.type()==6) {
			loadedImage.convertTo(loadedImage, 16);
			Imgproc.cvtColor(loadedImage, loadedImage, Imgproc.COLOR_GRAY2BGR);
		}
		mask.loadedImage				= loadedImage;		
		LinkedHashMap<ArrayList<Integer>,Double> pixelsInRect = mask.getRect(x, y, width, height);		
//		System.out.println(pixelsInRect);
		for (ArrayList<Integer> pixelPos : pixelsInRect.keySet()) {
			double [] p = {255,0,0};
			loadedImage.put(pixelPos.get(1), pixelPos.get(0), p);
		}
		showLoadedImageInUi(loadedImage);
	}
	public void test4() {
		path							= System.getProperty("user.dir").replace('\\', '/') + "/face00008.pgm";;
		loadedImage						= Imgcodecs.imread(path);
//		currentZoomFactor				=10.0f;
		System.out.println(loadedImage);
		showLoadedImageInUi(loadedImage);	
		
		localBinaryPattern();		
		
		
	}

	public void localBinaryPattern() {
		int maskSize					= getUserInt("3", "MaskSize", "3x3 5x5...etc", "Please Enter One Integer");
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, maskSize);
		mask.calculateRelPixelPositionsInMask();
//		System.out.println(mask.convolutionKernel);
//		getConvolutionKernelFromUI();
		mask.loadedImage				= loadedImage;
//		mask.convolutionKernel			= convolutionKernel;
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 0);
		
//		System.out.println(loadedImage.size().width);
		double newPixel;
		for (int col=0; col<loadedImage.size().width;col++) {
			for (int row=0; row<loadedImage.size().height;row++) {
				
//				System.out.println(loadedImage.get(row, col)[2]);
				mask.pixelX =col;
				mask.pixelY = row;
				mask.calculateAbsPixelPositionsInMask();
//				System.out.println(mask.pixelsUnderMask);
				newPixel				= mask.calculateLBPPixelValue();
				bufferImage.put(row, col, newPixel);
//				System.out.println(newPixel+"...."+bufferImage.get(row, col)[0]);
				
			}

		}
		if (ToBufferImage.isSelected()) {
			loadedImage					= bufferImage.clone();
		}		
		showLoadedImageInUi(bufferImage);			
	}
	public void test3() {
		loadEdgeDetectionImage();
		String [] kernel = {"-1","0","1","-2","0","2","-1","0","1"};
//		String [] kernel = {"1","2","1","0","0","0","-1","-2","-1"};
		
		fillKernel(kernel);
		getConvolutionKernelFromUI();
		System.out.println(convolutionKernel);
		System.out.println(convolutionKernel.size());
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.calculateRelPixelPositionsInMask();
		getConvolutionKernelFromUI();
		mask.loadedImage				= loadedImage;
		mask.convolutionKernel			= convolutionKernel;
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 6);
//		System.out.println(loadedImage.size().width);
		double newPixel;
		mask.pixelX =18;
		mask.pixelY = 14;
				
		mask.calculateAbsPixelPositionsInMask();
		System.out.println(mask.pixelsUnderMask);
		newPixel				= mask.calculateNewPixelValue();
		System.out.println(mask.pixelsUnderMask);
		System.out.println("NEW PIXEL ="+newPixel);
		double[] p					= {newPixel,newPixel,newPixel};
		double[]d ={255,0,0};
		bufferImage.put(14, 18, p);
		bufferImage.put(13, 18, d);
		showLoadedImageInUi(bufferImage);
		System.out.println(bufferImage.get(14, 18)[0]);
	}
	public void loadImageFromDisk() {
		currentZoomFactor				=10.0f;
        JFileChooser chooser = new JFileChooser();
        int returnName = chooser.showOpenDialog(null);
        
        if (returnName == JFileChooser.APPROVE_OPTION) {
            File f = chooser.getSelectedFile();
            if (f != null) {
                path = f.getAbsolutePath();
            }
        }else {
        	return;
        }
		effectIterations				=0;
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
		System.out.println(path);
		System.out.println(loadedImage);
		updateStatusText("Loaded Image From Disk !!!");
        
	}
	public void zoomIn() {
		currentZoomFactor			= (float) imgView.getScaleX();
		currentZoomFactor		   += 0.1f;
		imgView.setScaleX(currentZoomFactor);
		imgView.setScaleY(currentZoomFactor);		
	}
	public void zoomOut() {
		currentZoomFactor			= (float) imgView.getScaleX();
		currentZoomFactor		   -= 0.1f;
		imgView.setScaleX(currentZoomFactor);
		imgView.setScaleY(currentZoomFactor);	
		
	}
	
	
	public void fillKernel(String[]kernel) {
		for ( Node valueUI :ConvMaskGridP.getChildren()) {			
			if (valueUI.idProperty().getValue() != null) {
//				System.out.print(valueUI.idProperty().getValue()+" ");
				String id 			= valueUI.idProperty().getValue();				
				ArrayList<Integer> pixelpos = new ArrayList<Integer>();
				TextField tf		= (TextField)valueUI;
//				System.out.println(id);
				if (id.equals("Conv00TF")) {
					tf.setText(kernel[0]);
				}
				if (id.equals("Conv01TF")) {
					tf.setText(kernel[1]);
				}	
				if (id.equals("Conv02TF")) {
					tf.setText(kernel[2]);
				}	
				if (id.equals("Conv10TF")) {
					tf.setText(kernel[3]);
				}	
				if (id.equals("Conv11TF")) {
					tf.setText(kernel[4]);
				}
				if (id.equals("Conv12TF")) {
					tf.setText(kernel[5]);
				}				
				if (id.equals("Conv20TF")) {
					tf.setText(kernel[6]);
				}
				if (id.equals("Conv21TF")) {
					tf.setText(kernel[7]);
				}				
				if (id.equals("Conv22TF")) {
					tf.setText(kernel[8]);
				}				
				
			}
		}	
		
	}
	
	public void mouthBridge() {
		enhance();
		//Cheeks High lighter
		String [] kernel = {"0","-3","3","0","-3","3","0","-3","3"};
//		String [] kernel = {"-4","-4","-4","1","1","1","3","3","3"};
		fillKernel(kernel);
		applyConvolutionOperator();
	}		
	
	public void noseBridge() {
		enhance();
		//Cheeks High lighter
		String [] kernel = {"-1","-1","-1","0","0","0","1","1","1"};
//		String [] kernel = {"-4","-4","-4","1","1","1","3","3","3"};
		fillKernel(kernel);
		applyConvolutionOperator();
	}		
	
	public void foreheadPrepProcess() {
		enhance();
		//Cheeks High lighter
		String [] kernel = {"3","-3","0","3","-3","0","3","-3","0"};
		fillKernel(kernel);
		applyConvolutionOperator();
		
	}		
	public void cheekBonesPrepProcess() {
		enhance();
		//Cheeks High lighter
		String [] kernel = {"-2","-2","0","-2","0","2","0","2","2"};
		fillKernel(kernel);
		applyConvolutionOperator();
		
	}	
	public void invertImage() {
		for (int col=0; col<loadedImage.size().width;col++) {
			for (int row=0; row<loadedImage.size().height;row++) {
				double pixelIntensity = loadedImage.get(row, col)[0];
				if (pixelIntensity<0)pixelIntensity=0;
				if (pixelIntensity>255)pixelIntensity=255;
				loadedImage.put(row, col, 255-pixelIntensity);
				
			}		
		}
	}
	public void eyesPreprocess() {
		enhance();
		String [] kernel = {"0","2","0","-1","6","-1","0","1","0"};
		fillKernel(kernel);
		applyConvolutionOperator();
		invertImage();
		showLoadedImageInUi(loadedImage);
	}
	public void sobelX() {		
		String [] kernel = {"-1","0","1","-2","0","2","-1","0","1"};
		fillKernel(kernel);
		applyConvolutionOperator();
		imageBufA 					= bufferImage.clone();
	}	
	

	public void denoise() {		
		String [] kernel = {"1/9","1/9","1/9","1/9","1/9","1/9","1/9","1/9","1/9"};
		fillKernel(kernel);
		applyConvolutionOperator();
		imageBufA 					= bufferImage.clone();
	}	
	
	public void enhance() {		
		String [] kernel = {"0","-1","0","-1","5","-1","0","-1","0"};
		fillKernel(kernel);
		applyConvolutionOperator();
		imageBufA 					= bufferImage.clone();
	}	
	
	public void sobelY() {
		for ( Node valueUI :ConvMaskGridP.getChildren()) {			
			if (valueUI.idProperty().getValue() != null) {
//				System.out.print(valueUI.idProperty().getValue()+" ");
				String id 			= valueUI.idProperty().getValue();				
				ArrayList<Integer> pixelpos = new ArrayList<Integer>();
				TextField tf		= (TextField)valueUI;
//				System.out.println(id);
				if (id.equals("Conv00TF")) {
					tf.setText("-1");
				}
				if (id.equals("Conv01TF")) {
					tf.setText("-2");
				}	
				if (id.equals("Conv02TF")) {
					tf.setText("-1");
				}	
				if (id.equals("Conv10TF")) {
					tf.setText("0");
				}	
				if (id.equals("Conv11TF")) {
					tf.setText("0");
				}
				if (id.equals("Conv12TF")) {
					tf.setText("0");
				}				
				if (id.equals("Conv20TF")) {
					tf.setText("1");
				}
				if (id.equals("Conv21TF")) {
					tf.setText("2");
				}				
				if (id.equals("Conv22TF")) {
					tf.setText("1");
				}				
				
			}
		}
		applyConvolutionOperator();
		imageBufB							= bufferImage.clone();
	}
	
	public int getUserInt(String defaultValue, String title, String header, String context) {
		TextInputDialog dialog = new TextInputDialog(defaultValue);
		dialog.setTitle(title);
		dialog.setHeaderText(header);
		dialog.setContentText(context);

		// Traditional way to get the response value.
		Optional<String> result = dialog.showAndWait();
		if (result.isPresent()){
			return Integer.parseInt(result.get());
		}
		return 0;
	}
	public void sobelOperator() {
		TextInputDialog dialog = new TextInputDialog("30");
		dialog.setTitle("Edge Detection Threshold");
		dialog.setHeaderText("Threshold");
		dialog.setContentText("Intensity Threshold = ");

		// Traditional way to get the response value.
		Optional<String> result = dialog.showAndWait();
		if (result.isPresent()){
			
					
			sobelX();
	//		saveSnapShot();
			sobelY();
	//		saveSnapShot();
			for (int col=0; col<loadedImage.size().width;col++) {
				for (int row=0; row<loadedImage.size().height;row++) {
					
					double gX					= imageBufA.get(row, col)[0];
					double gY					= imageBufB.get(row, col)[0];
					double newPixel				= Math.sqrt((Math.pow(gX, 2)+Math.pow(gY, 2)));
					if (newPixel<Double.parseDouble(result.get())) {
						newPixel				= 0;
					}
					double[] p					= {newPixel,newPixel,newPixel};
					bufferImage.put(row, col, p);
					
				}
				
			}		
			showLoadedImageInUi(bufferImage);
		}
	}
	public void writeImage() {
		File f 								= new File(path);
		String fileName						= f.getName();
		File file = new File("edited_"+fileName);
		System.out.println(file.getAbsolutePath());
//		System.out.println(path.replaceFirst(fileName, ));
		Imgcodecs.imwrite(file.getAbsolutePath(), bufferImage);
	}
	public void reloadImageFromDisk() {
		effectIterations				=0;
//		path							="C:\\Users\\ibrahim\\Desktop\\faceDetection\\testImages\\test1.jpg";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
//		System.out.println(SourceImgGRP.getChildren());
//		System.out.println(loadedImage);
		updateStatusText("Reloaded Image From Disk !!!");
		
	}
	public void loadRandomFaceImage() {
		currentZoomFactor				=10.0f;
		String directory				=System.getProperty("user.dir").replace('\\', '/') + "/mit-cbcl-faces-balanced/train/face";
		File aDirectory 				= new File(directory);
		String[] filesInDir 			= aDirectory.list();
		int randomIndex					= new Random().nextInt(filesInDir.length);
		path							= directory +"/"+filesInDir[randomIndex];
		System.out.println(path);
		loadedImage						= Imgcodecs.imread(path);
//		System.out.println(loadedImage);
//		System.out.println(loadedImage.channels());
		
		previewFeatureTest();
		showLoadedImageInUi(loadedImage);
//			sobelY();
//			writeImage();
		
	}

	public void previewFeatureTest() {
		ToBufferImage.setSelected(true);		
		applyFeatureFromDropBox();
//		highlightRect(6,4,6,6);
//		highlightRect(7,1,4,8);
	}
	
	public void loadRandomNoneFaceImage() {
		String directory				=System.getProperty("user.dir").replace('\\', '/') + "/mit-cbcl-faces-balanced/train/non-face";
		File aDirectory 				= new File(directory);
		String[] filesInDir 			= aDirectory.list();
		int randomIndex					= new Random().nextInt(filesInDir.length);
		path							= directory +"/"+filesInDir[randomIndex];
		System.out.println(path);
		loadedImage						= Imgcodecs.imread(path);
//		System.out.println(loadedImage);
//		System.out.println(loadedImage.channels());		
		previewFeatureTest();
		showLoadedImageInUi(loadedImage);
		
	}	
	public void test() {
		String directory				="C:\\Users\\ibrahim\\Pictures\\Brian";
		File aDirectory 				= new File(directory);
		String[] filesInDir 			= aDirectory.list();
		for (String dir :filesInDir) {
			path						= directory +"\\"+dir;
			System.out.println(directory +"\\"+dir);
			loadedImage						= Imgcodecs.imread(path);
			showLoadedImageInUi(loadedImage);
			sobelY();
			writeImage();
			
		}
//		path							="C:\\Users\\ibrahim\\Desktop\\faceDetection\\Valve_original_(1).PNG";
//		loadedImage						= Imgcodecs.imread(path);
//		showLoadedImageInUi();
//		writeImage();
	}
	public void test2() {
		int x = 1;
		int y = 1;
		System.out.println("INTENSITY = "+loadedImage.get(y, x)[0]+","+loadedImage.get(y, x)[1]+","+loadedImage.get(y, x)[2]);
		
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.calculateRelPixelPositionsInMask();
		getConvolutionKernelFromUI();
		mask.loadedImage				= loadedImage;
		mask.convolutionKernel			= convolutionKernel;
		mask.pixelX = x;
		mask.pixelY = y;
		mask.calculateAbsPixelPositionsInMask();
		System.out.println(mask.pixelsUnderMask);
		double new_pixel					= mask.calculateNewPixelValue();
		System.out.println(mask.pixelsUnderMask);
		System.out.println(new_pixel);
		
	}
	public void showLoadedImageInUi(Mat imageToShow) {
		
		MatOfByte byteMat 				= new MatOfByte();
		Imgcodecs.imencode(".bmp", imageToShow	, byteMat);
		Image javafx_image				= new Image(new ByteArrayInputStream(byteMat.toArray()));
		imgView 						= new ImageView(javafx_image);
		imgView.setScaleX(currentZoomFactor);
		imgView.setScaleY(currentZoomFactor);
		SourceImgGRP.getChildren().clear();
		SourceImgGRP.getChildren().add(imgView);
		
	}
	public void loadEdgeDetectionImage() {	
		effectIterations				=0;
		currentZoomFactor				=1.0f;
		path							= System.getProperty("user.dir").replace('\\', '/') + "/test-pattern.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
//		System.out.println(SourceImgGRP.getChildren());
//		System.out.println(loadedImage);
		updateStatusText("Loaded test-pattern.tif!");
	}
	
	public void loadEnhanceImage() {	
		effectIterations				=0;
		currentZoomFactor				=1.0f;
		path							= System.getProperty("user.dir").replace('\\', '/') + "/blurry-moon.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
//		System.out.println(SourceImgGRP.getChildren());
//		System.out.println(loadedImage);
		updateStatusText("Loaded blurry-moon.tif!");
	}	
	
	public void loadHubbleImage() {	
		effectIterations				=0;
		currentZoomFactor				=1.0f;
		path							= System.getProperty("user.dir").replace('\\', '/') + "/hubble.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
//		System.out.println(SourceImgGRP.getChildren());
//		System.out.println(loadedImage);
		updateStatusText("Loaded hubble.tif!");
	}		
	
	public void loadDenoiseImage() {	
		effectIterations				=0;
		currentZoomFactor				=1.0f;
		path							= System.getProperty("user.dir").replace('\\', '/') + "/ckt-board-saltpep.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
//		System.out.println(SourceImgGRP.getChildren());
//		System.out.println(loadedImage);
		updateStatusText("Loaded ckt-board-saltpep.tif!");
	}	
	
	
	public void applyFeatureFromDropBox() {
		reloadImageFromDisk();
		if(ApplyFaceFeatureCB.isSelected()) {
			features.get(FeaturesDB.getValue()).run();
		}
		
		if (ShowFeatureRoiCB.isSelected()) {
			ArrayList<Integer> roi		= featuresRoIs.get(FeaturesDB.getValue());
			highlightRect(roi.get(0), roi.get(1), roi.get(2), roi.get(3));
		}
	}
	public void applyConvolutionOperator() {
		effectIterations				+=1;
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.calculateRelPixelPositionsInMask();
		getConvolutionKernelFromUI();
		mask.loadedImage				= loadedImage;
		mask.convolutionKernel			= convolutionKernel;
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 6);
		
		
		double newPixel;
		for (int col=0; col<loadedImage.size().width;col++) {
			for (int row=0; row<loadedImage.size().height;row++) {
				
//				System.out.println(loadedImage.get(row, col)[2]);
				mask.pixelX =col;
				mask.pixelY = row;
				mask.calculateAbsPixelPositionsInMask();
//				System.out.println(mask.pixelsUnderMask);
				newPixel				= mask.calculateNewPixelValue();
				double[] p					= {newPixel,newPixel,newPixel};
				bufferImage.put(row, col, newPixel);
				
			}

		}
		if (ToBufferImage.isSelected()) {
			loadedImage					= bufferImage.clone();
		}		
		showLoadedImageInUi(bufferImage);

		updateStatusText("Applied Convolution Kernel "+effectIterations+ " Times");
		
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
		double scale							= 5;
		Bounds bounds 							= ConvBox.getLayoutBounds();
		WritableImage image 					= new WritableImage(
	            (int) Math.round(bounds.getWidth() * scale),
	            (int) Math.round(bounds.getHeight() * scale));
		
		SnapshotParameters snapshotParams   	= new SnapshotParameters();
		snapshotParams.setFill(javafx.scene.paint.Color.rgb(40, 40, 40, 1));
		snapshotParams.setTransform(javafx.scene.transform.Transform.scale(scale, scale));
		
//	    WritableImage image2 					= TreeP.snapshot(snapshotParams,null);
    	
	    ImageView view 							= new ImageView(ConvBox.snapshot(snapshotParams, image));
	    File file = new File(fileName+".png");
	    
	    try {
	        ImageIO.write(SwingFXUtils.fromFXImage(view.getImage(), null), "png", file);
	    } catch (IOException e) {
	        
	    }
	}	
	public void saveSnapShot() {
		saveAsPng("Result_" + System.currentTimeMillis() );
	}
}

