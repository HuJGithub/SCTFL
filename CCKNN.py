import math
from random import *
import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData

class CCKNN(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        if len(self.label_df) > 1:
            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index


            fail_feature = np.array(self.feature_df[equal_one_index])

            ex_index=[]
            for temp in fail_feature:
                for i in range(len(temp)):
                    if temp[i] == 0:
                        ex_index.append(i)
            select_index=[]
            for i in range(len(self.feature_df.values[0])):
                if i not in ex_index:
                    select_index.append(i)

            select_index=list(set(select_index))
            sel_feature = self.feature_df.values.T[select_index].T
            columns = self.feature_df.columns[select_index]
            self.feature_df = pd.DataFrame(sel_feature, columns=columns)
            #print(self.feature_df.shape)


            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index

            pass_feature1 = np.array(self.feature_df[equal_zero_index])
            pass_feature2 = np.array(self.feature_df[equal_zero_index])
            fail_feature = np.array(self.feature_df[equal_one_index])
            while len(fail_feature) <= 20:
                fail_feature = np.vstack((fail_feature, fail_feature))

            allfeature = np.vstack((pass_feature1, fail_feature))
            print(allfeature.shape)
            fnum = len(fail_feature)
            pnum = len(pass_feature2)
            flabel = np.ones(fnum).reshape((-1, 1))
            plabel = np.zeros(pnum).reshape((-1, 1))
            label = np.vstack((plabel, flabel))




            deletelist=[]
            for i in range(len(pass_feature1)):
                distance = np.sum(np.power((allfeature - pass_feature1[i]), 2), axis=1)
                index = np.argsort(distance)[:5]
                n = 0
                for j in index:
                    if label[j] == 0:
                        n = n + 1
                    else:
                        n = n - 1
                #print(n)
                if n <= 0:
                    deletelist.append(i)

            #print(pass_feature2.shape)
            pass_feature2 = np.delete(pass_feature2, deletelist, axis=0)
            #print(pass_feature2.shape)
            compose_feature = np.vstack((pass_feature2, fail_feature))
            print(compose_feature.shape)
            fnum = len(fail_feature)
            pnum = len(pass_feature2)
            flabel = np.ones(fnum).reshape((-1, 1))
            plabel = np.zeros(pnum).reshape((-1, 1))
            compose_label = np.vstack((plabel, flabel))

            self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
            #print(self.label_df)
            self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)
            #print(self.feature_df)










            self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
            self.feature_df = pd.DataFrame(compose_feature, columns=columns, dtype=float)

            #self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
            #print(self.feature_df.shape)


