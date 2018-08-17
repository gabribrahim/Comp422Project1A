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
    
	private Main main;
	private DataSetsLoader myDataLoader 	= new DataSetsLoader();
	public int effectIterations				=0;
	public HashMap<String,XYChart.Series<Number,Number>> chartSeries = new HashMap<>();
	public ScatterChart<Number,Number> scatterChart ;
	public List<String> featuresList 		= new ArrayList<String>();
	public Mat loadedImage;
	public String path 						= null;
	public ImageView imgView;
	public Float currentZoomFactor 			= 1.0f;
	public HashMap<ArrayList<Integer>,Float> convolutionKernel		= new HashMap<ArrayList<Integer>,Float>();
	public Mat imageBufA;
	public Mat imageBufB;
	public Mat bufferImage;
	Map<String, Runnable> features = new HashMap<>();
	@SuppressWarnings("static-access")
	public void setMain(Main main) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//		FeaturesDP.getChildrenUnmodifiable().add();
		this.main		= main;

		FeaturesDB.getItems().addAll("featureA","featureB","featureC","symmetryX");
		FeaturesDB.setValue("featureA");
		features.put("featureA", () -> featureA());	
		features.put("featureB", () -> featureB());	
		features.put("featureC", () -> featureC());	
		
//		loadEdgeDetectionImage();
//		test();
//		applyConvolutionOperator();
//		System.exit(0);
		loadRandomFaceImage();
//		features.get("FeatureA").run();
//		System.exit(0);
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
	

	
	public void checkSymmetryX() {
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.loadedImage				= loadedImage;
		updateStatusText("SymmetryX score " +mask.calculateSymmetryX());
		
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
		mask.loadedImage				= loadedImage;		
		LinkedHashMap<ArrayList<Integer>,Double> pixelsInRect = mask.getRect(x, y, width, height);		
		double totalIntensity			= mask.computeAverageForRect(pixelsInRect);
		System.out.println(pixelsInRect);
		for (ArrayList<Integer> pixelPos : pixelsInRect.keySet()) {
			double [] p = {255,0,0};
			loadedImage.put(pixelPos.get(1), pixelPos.get(0), 255);
		}
		showLoadedImageInUi(loadedImage);
		updateStatusText("Average Intensity In Rect = "+totalIntensity);
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
	public void featureA() {
		//Cheeks High lighter
		String [] kernel = {"-1","-1","-1","0","0","0","1","1","1"};
//		String [] kernel = {"-4","-4","-4","1","1","1","3","3","3"};
		fillKernel(kernel);
		applyConvolutionOperator();
		
	}		
	public void featureC() {
		//Cheeks High lighter
		String [] kernel = {"-2","-2","0","-2","0","2","0","2","2"};
		fillKernel(kernel);
		applyConvolutionOperator();
		
	}	
	public void featureB() {
		String [] kernel = {"0","-1","0","-1","6","-1","0","1","0"};
		fillKernel(kernel);
		applyConvolutionOperator();
		
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
		System.out.println(SourceImgGRP.getChildren());
		System.out.println(loadedImage);
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
		System.out.println(loadedImage);
		System.out.println(loadedImage.channels());
		
		previewFeatureTest();
		showLoadedImageInUi(loadedImage);
//			sobelY();
//			writeImage();
		
	}

	public void previewFeatureTest() {
		ToBufferImage.setSelected(true);
		enhance();
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
		System.out.println(loadedImage);
		System.out.println(loadedImage.channels());		
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
		path							= System.getProperty("user.dir").replace('\\', '/') + "/test-pattern.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
		System.out.println(SourceImgGRP.getChildren());
		System.out.println(loadedImage);
		updateStatusText("Loaded test-pattern.tif!");
	}
	
	public void loadEnhanceImage() {	
		effectIterations				=0;
		path							= System.getProperty("user.dir").replace('\\', '/') + "/blurry-moon.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
		System.out.println(SourceImgGRP.getChildren());
		System.out.println(loadedImage);
		updateStatusText("Loaded blurry-moon.tif!");
	}	
	
	public void loadHubbleImage() {	
		effectIterations				=0;
		path							= System.getProperty("user.dir").replace('\\', '/') + "/hubble.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
		System.out.println(SourceImgGRP.getChildren());
		System.out.println(loadedImage);
		updateStatusText("Loaded hubble.tif!");
	}		
	
	public void loadDenoiseImage() {	
		effectIterations				=0;
		path							= System.getProperty("user.dir").replace('\\', '/') + "/ckt-board-saltpep.tif";
		loadedImage						= Imgcodecs.imread(path);
		showLoadedImageInUi(loadedImage);
		System.out.println(SourceImgGRP.getChildren());
		System.out.println(loadedImage);
		updateStatusText("Loaded ckt-board-saltpep.tif!");
	}	
	
	
	public void applyFeatureFromDropBox() {
		reloadImageFromDisk();
		features.get(FeaturesDB.getValue()).run();
	}
	public void applyConvolutionOperator() {
		effectIterations				+=1;
		ImageConvolutionMask mask 		= new ImageConvolutionMask(loadedImage, loadedImage.size().width, loadedImage.size().height, 3);
		mask.calculateRelPixelPositionsInMask();
		getConvolutionKernelFromUI();
		mask.loadedImage				= loadedImage;
		mask.convolutionKernel			= convolutionKernel;
		bufferImage						= new Mat(loadedImage.height(), loadedImage.width(), 6);
		
		System.out.println(loadedImage.size().width);
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

