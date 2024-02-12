from pyspark.sql import SparkSession
from pyspark.sql.functions import col,date_format

def init_spark():
  sql = SparkSession.builder\
    .appName("trip-app")\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

def main():

    file = "/opt/spark-data/ratings20.csv"
    sql,sc = init_spark()

    df = sql.read.load(file, format="csv", inferSchema="true", header=True)\
            .select("userid", "movieid", "rating")

    # Muestra los primeros 10 registros del DataFrame
    df.show(10)
  
if __name__ == '__main__':
  main()
