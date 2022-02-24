from pyspark.sql import SparkSession, DataFrame
from functools import reduce

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

def original_csv():
    spark = SparkSession.builder.appName('experiment').getOrCreate() 
    return spark.read.csv('indian_liver_patient.csv', header = True, inferSchema = True)
