from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
import cloudpickle
import pandas as pd


# Crear la sesión de Spark
spark = SparkSession.builder \
    .appName("GenreRecommendation") \
    .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar") \
    .getOrCreate()

# Definir la URL de la base de datos PostgreSQL
url = "jdbc:postgresql://demo-database:5432/movies"
# Especificar las credenciales
properties = {
    "user": "postgres",
    "password": "casa1234",
    "driver": "org.postgresql.Driver"
}

# Leer datos directamente desde la base de datos
movies = spark.read.jdbc(url=url, table="movies", properties=properties)

# Preprocesar los géneros para que estén en un formato adecuado para el modelo
tokenizer = RegexTokenizer(inputCol="genres", outputCol="words", pattern="\\|")
stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")

# Crear un modelo BisectingKMeans para agrupar los géneros
bisecting_kmeans = BisectingKMeans(k=1300, seed=1, featuresCol="features", predictionCol="genre_cluster")

# Crear el pipeline
pipeline = Pipeline(stages=[tokenizer, stop_words_remover, vectorizer, bisecting_kmeans])

# Ajustar el modelo
model = pipeline.fit(movies)

results = model.transform(movies)


# Convertir el DataFrame de Spark a Pandas DataFrame
results_pandas = results.toPandas()

# Guardar el Pandas DataFrame en un archivo pkl
with open('resultados.pkl', 'wb') as f:
    cloudpickle.dump(results_pandas, f)
    
# Género de prueba del usuario
user_genres = ['Romance']
# user_genres = ['Drama']


# Crear un DataFrame con el género del usuario
user_df = spark.createDataFrame([(0, " ".join(user_genres))], ["id", "genres"])

# Transformar el género del usuario usando el mismo pipeline
user_result = model.transform(user_df)

# Obtener el cluster asignado al género del usuario
user_cluster = user_result.select("genre_cluster").collect()[0]["genre_cluster"]

# Filtrar las películas en el mismo clúster que el género del usuario
recommended_movies = results.filter(col("genre_cluster") == user_cluster).select("title")

# Imprimir las recomendaciones
print(f"Recomendaciones para el género seleccionado:")
recommended_movies.show(truncate=False)

# Detener la sesión de Spark
spark.stop()
