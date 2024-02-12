from pyspark.sql import SparkSession

def init_spark():
    
    sql = SparkSession.builder\
      .appName("RecommenderSystem")\
      .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar") \
      .config("spark.executor.memory", "2g") \
      .config("spark.executor.cores", "2") \
      .config("spark.driver.memory", "2g") \
      .config("spark.driver.cores", "2g") \
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
    file = "/opt/spark-data/ratings25.csv"
    sql,sc = init_spark()

    df = sql.read.load(file, format="csv", inferSchema="true", header=True)  \
          .select("userid", "movieid", "rating") 

    # Filter invalid coordinates
    df.write \
          .jdbc(url=url, table="movilens", mode='append', properties=properties) 

    
if __name__ == '__main__':
  main()
