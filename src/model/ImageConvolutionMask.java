package model;

import java.sql.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;

import org.opencv.core.Mat;

public class ImageConvolutionMask {
	public double originalImageWidth;
	public double originalImageHeight;
	public Mat loadedImage;
	public int pixelX; //Position X of Mask On Loaded Image
	public int pixelY; //Position Y of Mask On Loaded Image
	public int size;
	public HashMap<ArrayList<Integer>,Float> convolutionKernel		= new HashMap<ArrayList<Integer>,Float>();
	public HashMap<ArrayList<Integer>,Float> pixelsUnderMask		= new HashMap<ArrayList<Integer>,Float>();
	public ImageConvolutionMask(Mat loadedImage,double width, double height, int size) {
		super();
		this.originalImageWidth = width;
		this.originalImageHeight = height;
		this.size = size-1;
	}
	
	
	public Float getKernelValue(int relPosX,int relPosY) {
		ArrayList <Integer>pixelPosArray = new ArrayList<>();
		pixelPosArray.add(relPosX);
		pixelPosArray.add(relPosY);
		
		return convolutionKernel.get(pixelPosArray);
		
	}
	public ArrayList<ArrayList<Integer>> calculateRelPixelPositionsInMask() {
		pixelX = 0;
		pixelY = 0;
		convolutionKernel.clear();
		ArrayList<ArrayList<Integer>> pixels 	= new ArrayList<>();
		//Add Center Pixel
		ArrayList<Integer> temp_pixel			= new ArrayList<>();
		int posX;
		int posY;
		for (int col=0;col<=size;col++) {
			for (int row=0;row<=size;row++) {
				posX = pixelX - (size/2)+ col;
				posY = pixelY - (size/2)+ row;
				temp_pixel.add(posX);
				temp_pixel.add(posY);
				pixels.add(temp_pixel);
				convolutionKernel.put((ArrayList<Integer>) temp_pixel.clone(), 0.0f);
				temp_pixel.clear();	
//				System.out.print(col+" ,");
//				System.out.print(row + " ");
//				System.out.print("[ " + posX);
//				System.out.print(",");
//				System.out.print(posY+" ],");
			}
		}
		return pixels;
		
	}
	
	public float map(double totalPixelValue, float in_min, float in_max, float out_min, float out_max)
	{
	  return (float) ((totalPixelValue - in_min) * (out_max - out_min) / (in_max - in_min) + out_min);
	}
	
	public double calculateMeanFilterValue() {
		double totalPixelValue	= 0.0f;
		for (ArrayList<Integer> pixel: pixelsUnderMask.keySet()) {
			float oldPixelValue = pixelsUnderMask.get(pixel);
			totalPixelValue += oldPixelValue;
//			pixelsUnderMask.put(pixel, newPixelValue);
		}		
		totalPixelValue			= totalPixelValue/pixelsUnderMask.size();
		return totalPixelValue;
	}
	
	public double calculateMedianFilterValue() {
		
		ArrayList <Integer> pixels = new ArrayList<>();		
		for (ArrayList<Integer> pixel:pixelsUnderMask.keySet()) {
//			pixelValues[counter]=Math.round(pixelsUnderMask.get(pixel));
			pixels.add(Math.round(pixelsUnderMask.get(pixel)));			
			}
		Object[]pixelValues = pixels.toArray() ;
		Arrays.sort(pixels.toArray());
		int middle = pixelValues.length/2;
		int medianValue = 0; //declare variable 
		if (pixelValues.length%2 == 1) { 
		    medianValue = (int) pixelValues[middle];
//			System.out.println(pixels+"\n"+medianValue);
		}
		else {
		   medianValue = ((int)pixelValues[middle-1] + (int)pixelValues[middle]) / 2;
		}
		return medianValue;
		
	}
	public ArrayList<Integer> range(int start, int stop)
	{
	   ArrayList<Integer> result 		= new ArrayList<>();

	   for(int i=0;i<stop-start;i++)
	      result.add(start+i);

	   return result;
	}	
	public ArrayList<ArrayList<Integer>> getSymmetryXhalves() {
		int middlePixel				= loadedImage.width()/2;
		int remainder				= loadedImage.width()%2;
		ArrayList<ArrayList<Integer>> imageHalves = new ArrayList<>();
		ArrayList<Integer> leftSide = range(0,middlePixel);
		ArrayList<Integer>rightSide;
		if (remainder==1) {
			rightSide				= range(middlePixel+1,loadedImage.width());	
		}else {
			rightSide				= range(middlePixel,loadedImage.width());
		}
		Collections.reverse(rightSide);
		System.out.println(remainder);
		System.out.println("Left  side column Count "+leftSide);
		System.out.println("Right side column Count "+rightSide);
		imageHalves.add(leftSide);
		imageHalves.add(rightSide);
		return imageHalves;
				
	}
	public double computeAverageForRect(LinkedHashMap<ArrayList<Integer>,Double> pixels) {
		double average					= 0;
		for (ArrayList<Integer> pixelPos :pixels.keySet()) {
			double pixelIntensity		= pixels.get(pixelPos);
			if (pixelIntensity<0)pixelIntensity=0;
			if (pixelIntensity>255)pixelIntensity=255;
			pixels.put(pixelPos, pixelIntensity);
			average						+= pixelIntensity;
		}
		System.out.print("Total Intensity = "+ average +" / "+pixels.size()+" = ");
		average							= average/pixels.size();
		System.out.println(average);
		return average;
	}
	public LinkedHashMap<ArrayList<Integer>,Double> getRect(int x, int y, int width, int height){
		 LinkedHashMap<ArrayList<Integer>,Double> pixels	= new LinkedHashMap<>();
		
		for (int col=x; col<=width+x;col++) {
			for (int row=y; row<=height+y;row++) {
				ArrayList<Integer> pixelPos = new ArrayList<>();
				pixelPos.add(col);
				pixelPos.add(row);
				pixels.put(pixelPos, loadedImage.get(row, col)[0]);
			}
		}
		return pixels;
	}
	public double calculateSymmetryX() {
		
		ArrayList<Double> leftSide	= new ArrayList<>();
		ArrayList<Double> rightSide	= new ArrayList<>();
		ArrayList<ArrayList<Integer>> imageHalves = getSymmetryXhalves();
		
		// GENERATING TWO SIDES LISTS TO COMPARE
		for (ArrayList<Integer> half: imageHalves) {
			int index				= imageHalves.indexOf(half);
			for (int col:half) {
				for (int row=0; row<loadedImage.size().height;row++) {
					if (index==0) {
						leftSide.add(loadedImage.get(row, col)[0]);
					}
					else {
						rightSide.add(loadedImage.get(row, col)[0]);
					}
				}
			}
		}
		// COMPARING TWO SIDES TO GENERATE AS SYMMETRY SCORE
		double symmetryScore 		= calculateSymmetryScoreForQuads(leftSide, rightSide);
//		System.out.println("Left Side "+leftSide.size());
//		System.out.println("Right Side "+rightSide.size());
		return symmetryScore;
	}


	public double calculateSymmetryScoreForQuads(ArrayList<Double> quad1, ArrayList<Double> quad2) {
		int pixelCounter			= 0;
		double symmetryScore		= 0;
		for (Double pixelIntensity:quad1) {
			Double rightPixelIntensity = quad2.get(pixelCounter);
			double diffIntensity	= 1 - (Math.abs(pixelIntensity-rightPixelIntensity)/255.0);
			diffIntensity			= diffIntensity/quad1.size();
			symmetryScore	       += diffIntensity;
//			System.out.println(pixelIntensity+"     "+rightPixelIntensity+" "+diffIntensity);
			pixelCounter++;
		}
		return symmetryScore;
	}
	public double calculateLBPPixelValue() {
		double middlePixelValue 	= pixelsUnderMask.get(Arrays.asList(0,0));
		String c					= "";
		for (ArrayList<Integer> pixel: pixelsUnderMask.keySet()) {
			if(pixel.equals(Arrays.asList(0,0)))continue;
			int newPixelValue		= 0;
//			System.out.println(pixel);
			float value = convolutionKernel.get(pixel);
			float oldPixelValue = pixelsUnderMask.get(pixel);
			if (oldPixelValue>=middlePixelValue) {
				newPixelValue		= 1;
			}
			c						+=newPixelValue;
			pixelsUnderMask.put(pixel, (float)newPixelValue);
		}		
		int decimalValue 			= Integer.parseInt(c, 2);
		return decimalValue;
	}
	public double calculateNewPixelValue() {
		double totalPixelValue	= 0.0f;
		for (ArrayList<Integer> pixel: pixelsUnderMask.keySet()) {
			float value = convolutionKernel.get(pixel);
			float oldPixelValue = pixelsUnderMask.get(pixel);
			float newPixelValue = value * oldPixelValue;
			totalPixelValue += newPixelValue;
			pixelsUnderMask.put(pixel, newPixelValue);
		}		
//		totalPixelValue			= map(totalPixelValue,-1000,1000,0,255);
		return totalPixelValue;
	}
	public Float getPixelValueFromImage (int absPosX, int absPosY) {
		try {
			if (absPosX>=loadedImage.width()) {
				absPosX = loadedImage.width()-1;
			}
			if (absPosX <=0) {
				absPosX	=0;
			}
			if (absPosY>=loadedImage.height()) {
				absPosY = loadedImage.height()-1;
			}
			if (absPosY<=0) {
				absPosY=0;
			}
			
			double[] pixel = loadedImage.get(absPosY,absPosX);
			
//			System.out.println(absPosX + " ," + absPosY + ">>"+Arrays.toString(pixel));
			return (float)loadedImage.get(absPosY,absPosX)[0];
		}
		catch (Exception ex) {
			System.out.println(absPosX + " OutofBounds " + absPosY);
//			System.out.println(ex);
			return 0.0f;
		}
	}
	
	@SuppressWarnings("unchecked")
	public ArrayList<ArrayList<Integer>> calculateAbsPixelPositionsInMask() {
		pixelsUnderMask.clear();
		ArrayList<ArrayList<Integer>> pixels 	= new ArrayList<>();
		//Add Center Pixel
		ArrayList<Integer> temp_pixel			= new ArrayList<>();
		int posX;
		int posY;
		int absPosX;
		int absPosY;		
		for (int col=0;col<=size;col++) {
			for (int row=0;row<=size;row++) {
				posX 							= 0 - (size/2)+ col;
				posY 							= 0 - (size/2)+ row;
				absPosX							= pixelX - (size/2)+ col;
				absPosY							= pixelY - (size/2)+ row;
				
				temp_pixel.add(posX);
				temp_pixel.add(posY);
				pixels.add(temp_pixel);
				pixelsUnderMask.put((ArrayList<Integer>) temp_pixel.clone(), getPixelValueFromImage(absPosX, absPosY) );
				temp_pixel.clear();	
			}
		}
		return pixels;
		
	}	
}
