from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
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
    table = "movilens"
    data_df = read_data_from_postgres(spark, url, properties, table)

   

    # Entrenar modelo ALS con Spark ML
    als = ALS(rank=10, maxIter=6, seed=70, userCol="userid", itemCol="movieid", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(data_df)


    # Contar el número de calificaciones para cada película
    movie_ratings_count = data_df.groupBy('movieid').count()

    # Filtrar películas con pocas calificaciones pero más de 4
    popular_movies = movie_ratings_count.filter((movie_ratings_count['count'] < 10) & (movie_ratings_count['count'] > 4))

    # Filtrar películas con altos ratings (mayores a 4)
    high_rated_movies = data_df.filter(data_df['rating'] > 4.0)

    # Realizar una operación JOIN para encontrar películas que cumplen ambas condiciones
    new_movies_for_recommendation = popular_movies.join(high_rated_movies, 'movieid', 'inner').select('movieid').distinct()
   
   # Generar recomendaciones para películas con poca participación pero altas calificaciones
    recommendations = model.recommendForItemSubset(new_movies_for_recommendation, 5)
   
    from pyspark.ml.evaluation import RegressionEvaluator
    # Mostrar las recomendaciones
    print(f"Top 5 Recommendations for Popular High-Rated Movies:")

    
    # Suponiendo que tienes un conjunto de datos de prueba llamado 'test_data_df'
    predictions = model.transform(data_df)

    # Configurando el evaluador para evaluar el RMSE
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    # Calculando el RMSE
    rmse = evaluator.evaluate(predictions)

    print(f"Root Mean Squared Error (RMSE) en el conjunto de prueba: {rmse}")
    # Mostrar las recomendaciones de manera mejorada
    print("Top 5 Recommendations for Popular High-Rated Movies:")
    for reco in recommendations.collect():
        movie_id = reco['movieid']
        
        print(f"Película ID: {movie_id}")
        print(f"Root Mean Squared Error (RMSE) en el conjunto de prueba: {rmse}", rmse)
      
        
        print()  # Línea en blanco para separar las recomendaciones

        



if __name__ == '__main__':
    main()
