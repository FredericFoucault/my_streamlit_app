#!/usr/bin/env python
# coding: utf-8



import unittest
import pandas as pd
import requests
import json
from sklearn.utils import resample



#df = pd.DataFrame({'A':[x for x in range(20450)]})
#df =pd.read_csv("application_cleaned.csv", sep=',',encoding="utf-8",low_memory=False).dropna(how='all', axis='columns')
data = requests.get('https://appprediction-b8add0149604.herokuapp.com/data').text
data = pd.DataFrame(json.loads(data))


df_majority = data[data.TARGET==0]
df_minority = data[data.TARGET==1]
df_test = data[data.TARGET=='NaN']

df_minority_upsampled = resample(df_minority, 
                                 replace=True,    #  remplacement
                                 n_samples=9225,     # to match majority class
                                 random_state=926) # reproducible results

df_upsampled = pd.concat([df_majority, df_minority_upsampled,df_test])


def long_data(x):
    """
    Teste le nombre de ligne de la table
    """
    lgd = len(x)
    return(lgd)


class TestProgram(unittest.TestCase):
    
    def test_long_data(self):
        self.assertEqual(10000, long_data(data))

def equal_target(x):
    """
    Teste l'egalité du upsampling des cibles('target')')
    """
    w = x.TARGET.value_counts()
    return(w)

class TestProgram(unittest.TestCase):
    
    def test_equal_target(self):
        self.assertEqual(equal_target(df_upsampled)[0], equal_target(df_upsampled)[1])


def more_F_than_M(x):
    """
    Teste que la composition du numbre de femme et d'homme n'est pas egal
    """
    w = x.CODE_GENDER.value_counts()
    return(w)

class TestProgram(unittest.TestCase):
    
    def test_equal_target(self):
        self.assertNotEqual(more_F_than_M(df_upsampled)[0], more_F_than_M(df_upsampled)[1])


if __name__ == "__main__":
  unittest.main()





