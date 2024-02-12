from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

def init_spark():
    spark = SparkSession.builder \
        .appName("trip-app") \
        .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar") \
        .getOrCreate()
    return spark

def main():
    file = "/opt/spark-data/u.data"
    spark = init_spark()

    # Cargar datos
    lines = spark.read.text(file)
    data_df = lines.rdd.map(lambda r: tuple(map(int, r.value.split('\t')[:3]))).toDF(["userid", "movieid", "ratingid"])

    # Calcular el promedio de rating por usuario y película
    consolidated_df = data_df.groupBy('userid', 'movieid').agg(avg('ratingid').alias('mean_rating'))

    # Pivotear los datos para crear una matriz usuario-película
    pivot_df = consolidated_df.groupBy('userid').pivot('movieid').agg(avg('mean_rating')).na.fill(0)

    # Definir la función de distancia manhattan correctamente
    def manhattan_distance(array1, array2):
        return sum(abs(a - b) for a, b in zip(array1, array2))

    # Definir el usuario objetivo
    target_user_id = 1

    # Filtrar datos del usuario objetivo de manera eficiente
    target_user_data = pivot_df.filter(col("userid") == target_user_id).select(*pivot_df.columns[1:]).collect()

    if not target_user_data:
        print(f"No hay datos para el usuario {target_user_id}")
    else:
        target_user_data = target_user_data[0]

    # Calcular distancias utilizando operaciones vectorizadas
    distances = pivot_df.filter(col("userid") != target_user_id).rdd.map(
        lambda row: (row.userid, manhattan_distance(row[1:], target_user_data))
    ).collect()

    # Ordenar las distancias
    sorted_distances = sorted(distances, key=lambda x: x[1])

    print(f"Vecinos más cercanos para el Usuario {target_user_id}: {sorted_distances}")

if __name__ == '__main__':
    main()
