from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def init_spark():
    sql = SparkSession.builder\
        .appName("trip-app")\
        .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar")\
        .getOrCreate()
    sc = sql.sparkContext
    return sql, sc

def main():
    url = "jdbc:postgresql://demo-database:5432/movilens"
    properties = {
        "user": "postgres",
        "password": "casa1234",
        "driver": "org.postgresql.Driver"
    }
    file = "/opt/spark-data/ratings25.csv"
    sql, sc = init_spark()

    df = sql.read.load(file, format="csv", inferSchema="true", header=True) \
        .select("userid", "movieid", "rating")

    # Verificar y eliminar datos duplicados basados en userid y movieid
    df = df.dropDuplicates(["userid", "movieid"])

    df.write \
        .jdbc(url=url, table="base20", mode='append', properties=properties)

if __name__ == '__main__':
    main()
