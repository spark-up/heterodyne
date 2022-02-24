from __future__ import annotations
import pandas as pd
import timeit

from src.heterodyne.features._impl import *

from pyspark.sql import functions as F
from pyspark.sql import SparkSession, DataFrame
from functools import reduce
from pyspark.context import SparkContext

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

spark = SparkSession.builder.appName('experiment').getOrCreate() 
sc = spark.sparkContext
sc.setCheckpointDir('checkpoint')
def original_csv():
    #spark = SparkSession.builder.appName('experiment').getOrCreate() 
    return spark.read.csv('indian_liver_patient.csv', header = True, inferSchema = True)

def scalability_testing_setup(n: int):
  def inner():
    df = original_csv()
    for i in range(n - 1):
      df = df.union(df)
    return df.checkpoint(True)
  return inner


def testing(i):
    TEST_CODE = '''
extract_features(df)
'''
    #return timeit.timeit(setup = SETUP_CODE, stmt = TEST_CODE, number = 1)
    return timeit.timeit('extract_features(df)', 'df=setup()', 
    globals={'setup': scalability_testing_setup(i), 'extract_features': extract_features}, number = 3)

def scalibility_testing(n):
    result_lst = []
    
#     SETUP_CODE = '''
# from src.heterodyne.features._impl import extract_features
# from scaling_resources import original_csv 
# df = original_csv()
# '''
    #SETUP_CODE = scalability_testing_setup(7)

    for i in range(n):
        result_lst.append(testing(i))
        #SETUP_CODE = SETUP_CODE + '\ndf = df.union(df) \ndf = df.checkpoint(eager=True)'

    return result_lst

print(scalibility_testing(7))

print(pd.read_csv('indian_liver_patient.csv'))
