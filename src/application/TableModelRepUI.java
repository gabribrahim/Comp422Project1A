package application;

import javafx.beans.property.SimpleStringProperty;

public class TableModelRepUI {

    public SimpleStringProperty className = new SimpleStringProperty();
    public SimpleStringProperty spam  = new SimpleStringProperty();
    public SimpleStringProperty NoSPam  = new SimpleStringProperty();
    public SimpleStringProperty prob  = new SimpleStringProperty();
	public TableModelRepUI(String className, String spam, String noSpam,
			String prob) {
		super();
		this.className.set(className);
		this.spam.set(spam);
		NoSPam.set(noSpam);
		this.prob.set(prob);
	}
	public String getSpam() {
		return spam.get();
	}
	public void setSpam(SimpleStringProperty spam) {
		this.spam = spam;
	}
	public String getNoSPam() {
		return NoSPam.get();
	}
	public void setNoSPam(SimpleStringProperty noSPam) {
		NoSPam = noSPam;
	}
	public String getProb() {
		return prob.get();
	}
	public void setProb(SimpleStringProperty prob) {
		this.prob = prob;
	}
	public String getClassName() {
		return className.get();
	}
	public void setClassName(String className) {
		this.className.set(className);
	}
    

}