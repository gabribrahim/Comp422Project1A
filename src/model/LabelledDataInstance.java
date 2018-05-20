package model;


import java.util.ArrayList;
import java.util.List;

public class LabelledDataInstance <LabelledDataInstance>{
	public List<String> featuresListAsStrings;
	public String labelName;
	public ArrayList<Boolean> featureListAsValues;	
	public String predictedClass;
	public LabelledDataInstance(List<String> featuresList, String labelName) {	
		super();
		this.featuresListAsStrings 			= featuresList;
		this.labelName 						= labelName;
		
	}
	
	@Override
	public String toString() {
		return "LabelledDataInstance [labelName=" + labelName + ", featureListAsValues=" + featureListAsValues + " "
				+"]";
	}

	public void parseInformationToValues() {
		featureListAsValues					= new ArrayList<Boolean>();
		for (String string : featuresListAsStrings) {
//			System.out.println(string);
			if(string.contains("0")) {
				featureListAsValues.add(false);
			}else if(string.contains("1")) {
				featureListAsValues.add(true);
			}else {
				featureListAsValues.add(false);
			}
			
		}
	}




	

}
