package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;


public class DataSetsLoader {
	public ArrayList<LabelledDataInstance> trainingDataSetList= new ArrayList<LabelledDataInstance>();
	ArrayList<Float> minValuesForFeatures			= new ArrayList<Float>();
	ArrayList<Float> maxValuesForFeatures			= new ArrayList<Float>();
	public ArrayList<LabelledDataInstance> testDataSetList= new ArrayList<LabelledDataInstance>();
	public HashSet<String> dataSetClasses			= new HashSet<>();
	public BaseClassifier myclassifier;
	public String dataSetName;
	public HashMap<String,Integer> classesCount		= new HashMap<>();
	public LinkedHashMap<String,HashMap<String,Integer>> countTable	= new LinkedHashMap<>();
	public LinkedHashMap<String,HashMap<String,String>> probTable	= new LinkedHashMap<>();
	@Override
	public String toString() {
		String strRep								= dataSetName + " [dataSetListSize=" + trainingDataSetList.size() + "]\n";
		strRep									   += "               [MinimumRange = " + minValuesForFeatures + "]\n";
		strRep									   += "               [MaximumRange = " + maxValuesForFeatures + "]\n";
		strRep									   += "               [DataSetLabels= " + dataSetClasses + "]\n";
		strRep									   += "               [LabelsCount  = " + dataSetClasses.size() + "]\n";

		return strRep;
	}

	
	public void clear() {
		this.trainingDataSetList.clear();
		this.testDataSetList.clear();
		this.minValuesForFeatures.clear();
		this.maxValuesForFeatures.clear();
	}
	
	public void clearRanges() {
		this.minValuesForFeatures.clear();
		this.maxValuesForFeatures.clear();
		
	}
	public String loadEmailData(String filePath, ArrayList<LabelledDataInstance> dataSetList, Boolean dataIslabelled) {		
		
		File fileObj 								= new File(filePath);
		//Reading File Contents & Creating Java Representation Objects for DataSet For Classifier Classes//
		try (FileReader fileReader = new FileReader(fileObj);
				BufferedReader bufferedReader		= new BufferedReader(fileReader);){
			String line 							= bufferedReader.readLine();
			
			while (line != null) {
				String[] lineParts 					= line.split("     ");
				List<String> lineList 				= new LinkedList<String>(Arrays.asList(lineParts));				
				lineList.remove("");
				String featureLabel;
				List<String> featuresList;
				if (dataIslabelled) {
					featureLabel 					= lineList.get(lineList.size() - 1);
					featuresList 					= lineList.subList(0, lineList.size() - 1);
				}else {
					featureLabel 					= "";
					featuresList 					= lineList.subList(0, lineList.size());
				}
//				System.out.println(featuresList);
				LabelledDataInstance dataInstance 	= new LabelledDataInstance(featuresList, featureLabel);
				dataSetClasses.add(featureLabel);
				dataInstance.parseInformationToValues(); // Converts Strings to Floats
				dataSetList.add(dataInstance);
//				System.out.println(dataInstance+"\n"+featuresList.size()+"<>"+dataInstance.featureListAsValues.size());
				
				line 								= bufferedReader.readLine();
			}			
		} catch (IOException e) {
			System.out.println("FILE NOT FOUND !!");
		}
		
		String report								= toString();
		return report;
	}

	public void createCountTable() {
		classesCount.put("1", 0);
		classesCount.put("0", 0);
		countTable.put("Total",new HashMap<String,Integer>());
		for (int i=0;i<trainingDataSetList.get(0).featureListAsValues.size();i++) {
			HashMap<String,Integer> labelsMap = new HashMap<String,Integer>();
			for (String classLabel : dataSetClasses) {
				labelsMap.put(classLabel, 0);
			}
			countTable.put("Feature"+i+"=true",labelsMap);
			countTable.put("Feature"+i+"=false",(HashMap<String, Integer>) labelsMap.clone());
		}
		for (LabelledDataInstance instance:trainingDataSetList) {
			//COUNT CLASS LABELS
			int currentCount 						= classesCount.get(instance.labelName);
			currentCount							+=1;
			classesCount.put(instance.labelName, currentCount);
			// Features Count 
			for (int i=0;i<instance.featureListAsValues.size();i++) {
				
				boolean state						= instance.featureListAsValues.get(i);
				if (state) {
					int featureCount				= countTable.get("Feature"+i+"=true").get(instance.labelName);
					featureCount				   +=1;
					countTable.get("Feature"+i+"=true").put(instance.labelName, featureCount);
				}else {
					int featureCount				= countTable.get("Feature"+i+"=false").get(instance.labelName);
					featureCount				   +=1;
					countTable.get("Feature"+i+"=false").put(instance.labelName, featureCount);
					
				}
				
				
			}
			
		}
		
		countTable.get("Total").put("1", classesCount.get("1"));
		countTable.get("Total").put("0", classesCount.get("0"));
	}
	
	public void createProbTable() {
		// Below is Just String Reps For UI AND Sanity Visual Checks
		// Filling Up Totals 
		probTable.put("P(Class)",new HashMap<String,String>());
		probTable.get("P(Class)").put("1", countTable.get("Total").get("1").toString()+"/"+trainingDataSetList.size());
		probTable.get("P(Class)").put("0", countTable.get("Total").get("0").toString()+"/"+trainingDataSetList.size());
		
		// Creating Empty Maps For All feature=true , feature=false
		for (int i=0;i<trainingDataSetList.get(0).featureListAsValues.size();i++) {
			HashMap<String,String> labelsMap = new HashMap<String,String>();
			for (String classLabel : dataSetClasses) {
				labelsMap.put(classLabel, "");
			}
			probTable.put("Feature"+i+"=true",labelsMap);
			probTable.put("Feature"+i+"=false",(HashMap<String, String>) labelsMap.clone());
		}
		// Building Strings count(feature=true)/count(class), count(feature=false)/count(class)
		for (Entry<String, HashMap<String, Integer>> entry:countTable.entrySet()) {
			if (entry.getKey()=="Total") {continue;}
			for (String classLabel:dataSetClasses) {
				int featureCount 				= countTable.get(entry.getKey()).get(classLabel);
				int totalCount					= countTable.get("Total").get(classLabel);				
				probTable.get(entry.getKey()).put(classLabel, ""+featureCount+"/"+totalCount);
				
			}
			
		}
	}
	
	
	
	
	
	public String loadIrisDataSet(String filePath, ArrayList<LabelledDataInstance> dataSetList) {		
		
		File fileObj 								= new File(filePath);
		//Reading File Contents & Creating Java Representation Objects for DataSet For Classifier Classes//
		try (FileReader fileReader = new FileReader(fileObj);
				BufferedReader bufferedReader		= new BufferedReader(fileReader);){
			String line 							= bufferedReader.readLine();
			
			while (line != null) {
				String[] lineParts 					= line.split("  ");
				if (lineParts.length < 5) {
					break;
				}				

				List<String> lineList 				= Arrays.asList(lineParts);
				String featureLabel 				= lineList.get(lineList.size() - 1);
				List<String> featuresList 			= lineList.subList(0, 4);
				LabelledDataInstance dataInstance 	= new LabelledDataInstance(featuresList, featureLabel);
				dataSetClasses.add(featureLabel);
				dataInstance.parseInformationToValues(); // Converts Strings to Floats
				dataSetList.add(dataInstance);
				line 								= bufferedReader.readLine();
			}			
		} catch (IOException e) {
			System.out.println("FILE NOT FOUND !!");
		}
		
		String report								= toString();
		return report;
	}
}
