from pyspark.sql import SparkSession
from pyspark.sql.functions import col,date_format

def init_spark():
  sql = SparkSession.builder\
    .appName("trip-app")\
    .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar")\
    .config("spark.executor.memory", "1g") \
    .config("spark.executor.cores", "1") \
    .config("spark.driver.memory", "1g") \
    .config("spark.driver.cores", "1g") \
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

def main():
  url = "jdbc:postgresql://demo-database:5432/movilens"
  properties = {
    "user": "postgres",
    "password": "casa1234",
    "driver": "org.postgresql.Driver"
  }
  file = "/opt/spark-data/u.data"
  sql,sc = init_spark()

  df = sql.read.load(file, format="csv", inferSchema="true", sep="\t", header=False) \
        .select(col("_c0").alias("userid"), col("_c1").alias("movieid"), col("_c2").alias("rating"))


  # Filter invalid coordinates
  df.write \
        .jdbc(url=url, table="prueba", mode='append', properties=properties) 

  
if __name__ == '__main__':
  main()
