package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import javax.swing.plaf.synth.SynthSeparatorUI;

import javafx.scene.control.Label;

public class KMeansClassifier extends BaseClassifier{
	public DataSetsLoader myDataSet;
	public int kNumber							= 3;
	public HashMap<String,ArrayList<LabelledDataInstance>> clusterMap			= new HashMap<String,ArrayList<LabelledDataInstance>>();
	public ArrayList<LabelledDataInstance> clusterPoints						= new ArrayList<>();
	public HashMap<String,ArrayList<Float>> centroidMeans									= new HashMap<>();
	public KMeansClassifier(DataSetsLoader myDataSet) {
		super();
		this.myDataSet = myDataSet;
	}
	
	public void initialiseRandomClusters() {
		Random randomPicker						= new Random();
		for (int i=0; i<kNumber; i++) {
			ArrayList<LabelledDataInstance> nearestPoints		= new ArrayList<>();
			LabelledDataInstance randomInstance = myDataSet.trainingDataSetList.get(randomPicker.nextInt(myDataSet.trainingDataSetList.size()));
			LabelledDataInstance cluster		= new LabelledDataInstance(randomInstance.featuresListAsStrings, "Cluster"+String.valueOf(i));
			cluster.parseInformationToValues();
			clusterMap.put(cluster.labelName,nearestPoints);
			clusterPoints.add(cluster);
			
			
		}
	}
	
	public String reportOnClusters() {
		String report							= "";
		for (LabelledDataInstance cluster: clusterPoints) {
			report								+= cluster.labelName +" Contains "+ clusterMap.get(cluster.labelName).size()+"Instances\n";
		}
		return report;
	}
	
	public void updateCentroids() {
		for (LabelledDataInstance cluster : clusterPoints) {
			centroidMeans.put(cluster.labelName, new ArrayList<Float>(cluster.featureListAsValues));
		}		
	}
	
	public boolean checkForConvergence() {
		boolean converged						= true;
		for (LabelledDataInstance cluster : clusterPoints) {
			if (centroidMeans.get(cluster.labelName)==null) {return false;}
			for (float mean :centroidMeans.get(cluster.labelName)) {
				int index						= centroidMeans.get(cluster.labelName).indexOf(mean);
				if (mean != cluster.featureListAsValues.get(index)) {return false;}
			}
		}
		return converged;
	}
	public void recomputerCentroids() {
		for (LabelledDataInstance cluster : clusterPoints) {
			for (int featureIndex = 0; featureIndex < cluster.featureListAsValues.size(); featureIndex++) {
				float averageOfFeature			= (float) 0.0;
				for (LabelledDataInstance trainingInstance: clusterMap.get(cluster.labelName)) {					
					averageOfFeature 			+= trainingInstance.featureListAsValues.get(featureIndex);
				}
				averageOfFeature 				= averageOfFeature / (float)clusterMap.get(cluster.labelName).size();
				cluster.featureListAsValues.set(featureIndex, averageOfFeature);				
			}
			
		}
	}
	public void clearClusterMap() {
		for (LabelledDataInstance cluster : clusterPoints) {
			clusterMap.get(cluster.labelName).clear();
		}
	}
	public void updateCluster() {		
		for (LabelledDataInstance trainingInstance : myDataSet.trainingDataSetList) {
			for (LabelledDataInstance cluster : clusterPoints) {
				double euclideanNormalisedDistance = 0.0;
				for (int featureIndex = 0; featureIndex < trainingInstance.featureListAsValues.size(); featureIndex++) {
					double featureRangeSquared = Math.pow(myDataSet.maxValuesForFeatures.get(featureIndex)
							- myDataSet.minValuesForFeatures.get(featureIndex), 2);
	
					double offsetBetweenFeatures = Math.pow(trainingInstance.featureListAsValues.get(featureIndex)
							- cluster.featureListAsValues.get(featureIndex), 2);
					euclideanNormalisedDistance += Math.sqrt(offsetBetweenFeatures / featureRangeSquared);					
					}
				cluster.euclideanNormalisedDistance = euclideanNormalisedDistance;
				}
			
			LabelledDataInstance nearestCluster		= clusterPoints.get(0);
//			System.out.println("Before Sort" + clusterPoints.get(0));
			Collections.sort(clusterPoints);
//			System.out.println("After Sort" + clusterPoints.get(0));
//			System.out.println("Adding Instance To "+ nearestCluster.labelName);
			clusterMap.get(nearestCluster.labelName).add(trainingInstance);
			
		}
		recomputerCentroids();
	}
	

}
