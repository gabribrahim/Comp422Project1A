package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

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
	
	public double calculateNewPixelValue() {
		double totalPixelValue	= 0.0f;
		for (ArrayList<Integer> pixel: pixelsUnderMask.keySet()) {
			float value = convolutionKernel.get(pixel);
			float oldPixelValue = pixelsUnderMask.get(pixel);
			float newPixelValue = value * oldPixelValue;
			totalPixelValue += newPixelValue;
			pixelsUnderMask.put(pixel, newPixelValue);
		}		
		return totalPixelValue;
	}
	public Float getPixelValueFromImage (int absPosX, int absPosY) {
		try {
			if (absPosX>loadedImage.width() |absPosX<0) {
				return 0.0f;
			}
			if (absPosY>loadedImage.height() |absPosY<0) {
				return 0.0f;
			}
			
			double[] pixel = loadedImage.get(absPosY,absPosX);
			
//			System.out.println(absPosX + " ," + absPosY + ">>"+Arrays.toString(pixel));
			return (float)loadedImage.get(absPosY,absPosX)[0];
		}
		catch (Exception ex) {
//			System.out.println(absPosX + " NOT WORKING " + absPosY);
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
