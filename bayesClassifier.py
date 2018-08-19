import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import time

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class SimpleBayesClassifier(object):
    def __init__(self):
        self.gnb = GaussianNB()
        self.filters_to_keep         = []
        self.filters_to_remove       = []
        self.features_importance_dict= {}
    def load_csv_files(self):
        # Importing dataset
        training_data  = pd.read_csv("trainingFace.csv")
        training_data2 = pd.read_csv("trainNoneFace.csv")
        training_data  = pd.concat([training_data,training_data2], ignore_index=True)
        
        test_data      = pd.read_csv("testFace.csv")
        test_data2     = pd.read_csv("testNoneFace.csv")
        test_data      = pd.concat([test_data,test_data2], ignore_index=True)
        
        self.training_data = training_data
        self.test_data = test_data
        #print training_data.shape[0],test_data.shape[0]
        #print training_data.axes
        
    def used_features(self):
        used_features = [u'symmetryX', u'noseUnfilteredAverage',
                         u'noseUnfilteredStd', u'noseUnfilteredMomentum',
               u'cheekBonesLeftUnfilteredAverage', u'cheekBonesLeftUnfilteredStd',
               u'cheekBonesLeftUnfilteredMomentum',
               u'cheekBonesRightUnfilteredAverage', u'cheekBonesRightUnfilteredStd',
               u'cheekBonesRightUnfilteredMomentum', u'eyeLeftUnfilteredAverage',
               u'eyeLeftUnfilteredStd', u'eyeLeftUnfilteredMomentum',
               u'eyeRightUnfilteredAverage', u'eyeRightUnfilteredStd',
               u'eyeRightUnfilteredMomentum', u'mouthUnfilteredAverage',
               u'mouthUnfilteredStd', u'mouthUnfilteredMomentum',
               u'foreheadUnfilteredAverage', u'foreheadUnfilteredStd',
               u'foreheadUnfilteredMomentum', u'noseAverage', u'noseStd',
               u'noseMomentum', u'cheekBonesLeftAverage', u'cheekBonesLeftStd',
               u'cheekBonesLeftMomentum', u'cheekBonesRightAverage',
               u'cheekBonesRightStd', u'cheekBonesRightMomentum', u'leftEyeAverage',
               u'rightEyeAverage', u'mouthAverage', u'mouthStd', u'mouthMomentum',
               u'foreheadAverage', u'foreheadStd', u'foreheadMomentum']
        
        filters_to_use              = self.filter_features(used_features)
        #print len(used_features),len(filters_to_use)
        
        return filters_to_use
    
    def filter_features(self,used_filters, keep_filters=None,remove_filters=None):
        filters_to_use              = []
        filters_to_remove           = []
        keep_filters                = self.filters_to_keep
        remove_filters              = self.filters_to_remove
        
        for filter_name in used_filters:
            for filter_frag in keep_filters:
                if filter_frag == "all":
                    filters_to_use.append(filter_name)
                    break
                if filter_frag in filter_name and filter_name not in filters_to_use:
                    filters_to_use.append(filter_name)
        
        for filter_name in filters_to_use:
            for filter_frag in remove_filters:
                if filter_frag in filter_name:
                    filters_to_remove.append(filter_name)
        
        #print filters_to_remove
        
        for filter_to_remove in filters_to_remove:
            #print filter_to_remove,filter_to_remove in filters_to_use
            if filter_to_remove in filters_to_use:
                filters_to_use.remove(filter_to_remove)
                
        return filters_to_use 
    
    def feature_selection(self,features_importance_print=False):
        clf = ExtraTreesClassifier()        
        features_available = self.used_features()
        clf.max_features = len(features_available)
        clf = clf.fit(self.training_data[features_available].values, self.training_data["label"])
        for importance in clf.feature_importances_:
            index = np.where(clf.feature_importances_==importance)[0][0]
            self.features_importance_dict[importance] =features_available[index]
        
        if features_importance_print:
            for feature_score in sorted(self.features_importance_dict.keys(),reverse=True):
                print self.features_importance_dict[feature_score],feature_score
            
    
    def predict_using_top_feature_performers(self,topNperformers):
        self.feature_selection()
        features_to_use = []
        performers_count = 0
        for score in sorted(self.features_importance_dict.keys(),reverse=True):
            features_to_use.append(self.features_importance_dict[score])
            performers_count +=1
            if performers_count == topNperformers:
                break
        
        self.predict(features_to_use, self.training_data, self.test_data)
        
            
    def predict_all(self):
        self.predict(self.used_features(), self.training_data, self.test_data)
       
    def report_per_feature_error_rate(self):
        features_to_use            = self.used_features()
        for feature_name in features_to_use:
            self.filters_to_keep    = [feature_name]
            self.predict_all()
            
    def predict(self, features_to_use, training_set, test_set):
        self.model    = self.gnb.fit(training_set[features_to_use].values, training_set["label"])
        y_pred        = self.model.predict(test_set[features_to_use])
        error_rate    = 100*(float((test_set["label"] != y_pred).sum())/float(test_set.shape[0]))
        
        print "Total Tested", test_set.shape[0]
        print "Errors",(test_set["label"] != y_pred).sum(),error_rate
        line_frags = []
        line_frags.append(str((test_set["label"] != y_pred).sum()))
        line_frags.append(str(error_rate))
        line_frags.append(str(len(features_to_use)))
        line_frags.append(str(self.filters_to_keep).replace(","," "))
        line_frags.append(str(features_to_use))
        with open("report.csv", mode='a') as W:
            W.write(",".join(line_frags)+"\n")        
    
    def predict_each(self):
        self.model    = self.gnb.fit(self.training_data [self.used_features].values, self.training_data["label"])
        total_wrong_predictions = 0
        wrong_images_detected   = []
        for index,row in self.test_data.iterrows():
            predicition = self.model.predict([row[used_features]])[0]
            #print index,row['label'],predicition
            break
            if row['label']!=predicition:
                total_wrong_predictions +=1
                wrong_images_detected.append(row["filePath"])
                
        with open("wrongPredictions,txt",'w') as W:
            W.write("\n".join(wrong_images_detected))
        print total_wrong_predictions    


classifier =SimpleBayesClassifier()
classifier.load_csv_files()
classifier.filters_to_keep = ["all"]
#classifier.report_per_feature_error_rate()
classifier.predict_using_top_feature_performers(8)
#classifier.filters_to_remove = ["Unfilter","Std"]
#classifier.predict_all()
#classifier.feature_selection()


"""
gnb = GaussianNB()
# Importing dataset
training_data  = pd.read_csv("training_data /train.csv")
used_features =["Fare"]
y_pred = gnb.fit(X_train[used_features].values, X_train["Survived"]).predict(X_test[used_features])
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))
print("Std Fare not_survived {:05.2f}".format(np.sqrt(gnb.sigma_)[0][0]))
print("Std Fare survived: {:05.2f}".format(np.sqrt(gnb.sigma_)[1][0]))
print("Mean Fare not_survived {:05.2f}".format(gnb.theta_[0][0]))
print("Mean Fare survived: {:05.2f}".format(gnb.theta_[1][0]))
"""