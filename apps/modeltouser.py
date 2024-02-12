from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

# Función para inicializar Spark
def init_spark():
    spark = SparkSession.builder\
        .appName("RecommenderSystem")\
        .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar")\
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .config("spark.driver.cores", "2g") \
        .getOrCreate()
    return spark


def save_als_model(model, model_path):
    # Guardar el modelo ALS en formato nativo de PySpark
    model.write().overwrite().save(model_path)
    print(f"Modelo ALS guardado en formato nativo de PySpark en: {model_path}")



# Función para leer datos desde PostgreSQL y convertir a DataFrame de Spark
def read_data_from_postgres(spark, url, properties, table):
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    return df

def main():
    # Configuración de PostgreSQL
    url = "jdbc:postgresql://demo-database:5432/movilens"
    properties = {
        "user": "postgres",
        "password": "casa1234",
        "driver": "org.postgresql.Driver"
    }

    # Inicializar Spark
    spark = init_spark()

    # Leer datos desde PostgreSQL
    table = "data20"
    data_df = read_data_from_postgres(spark, url, properties, table)

    # Dividir datos en conjuntos de entrenamiento y prueba
    (training, test) = data_df.randomSplit([0.8, 0.2])

    # Entrenar modelo ALS con Spark ML
    als = ALS(rank=10, maxIter=6, seed=70, userCol="userid", itemCol="movieid", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)

    # Hacer recomendaciones para el usuario 1
    user_id_to_predict = 1

    # Crear un DataFrame con una sola fila para el usuario a predecir
    single_user_df = spark.createDataFrame([(user_id_to_predict,)], ["userid"])

    # Hacer recomendaciones para el usuario
    recommendations_user_1 = model.recommendForUserSubset(single_user_df, 10)

    # Mostrar las recomendaciones
    print(f"Top 10 Recommendations for User {user_id_to_predict}:")
    for i, reco in enumerate(recommendations_user_1.collect()[0]['recommendations'], 1):
        print(f"{i}. Película ID: {reco['movieid']}, Score: {reco['rating']}")
    
    # Guardar el modelo ALS en formato nativo de PySpark
    model_path = "/opt/spark-models/recommendation_model"
    save_als_model(model, model_path)

    print("-------------------")
    print("-------------------")


if __name__ == '__main__':
    main()
