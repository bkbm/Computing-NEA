# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:42:30 2018

@author: baile
"""

import numpy as np

class Data:
    Features = None
    Results = None
    NumberOfFeatures = None
    PossibleResults = None
    SizeOfData = None
    def __init__(self,features,results,possibleResults):       
        if len(features) == len(results):
            self.Features = features
            self.Results = results
        else:
            raise NameError('The number of rows in features needs to match that of results')
        if len(features.shape) == 1:
            self.NumberOfFeatures = 1
        else:
            self.NumberOfFeatures = features.shape[1]
        self.PossibleResults = possibleResults
        self.SizeOfData = len(features)
    def FormatResults(self):
        Categories = np.eye(self.PossibleResults)
        FormattedResults = np.zeros((self.SizeOfData,self.PossibleResults))
        for ix in range(0, self.SizeOfData):
            FormattedResults[ix,:] = Categories[self.Results[ix]-1,:]
        return FormattedResults

