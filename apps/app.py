from flask import Flask, jsonify ,request
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
app = Flask(__name__)

# Inicializar Spark como objeto global
spark = SparkSession.builder \
    .appName("RecommenderSystem") \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.memory", "1g") \
    .config("spark.cores.max", "1") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .getOrCreate()

# Leer datos desde PostgreSQL y obtener los primeros 5 registros
def get_first_5_records():
    table = "data20"
    properties = {
        "user": "postgres",
        "password": "casa1234",
        "driver": "org.postgresql.Driver"
    }
    df = spark.read.jdbc(url="jdbc:postgresql://demo-database:5432/movilens", table=table, properties=properties)
    first_5_records = df.limit(5).collect()
    return first_5_records


model_path = "/opt/spark-models/recommendation_model"

# Cargar el modelo ALS
model2 = ALSModel.load(model_path)


# Ruta de Flask para mostrar los primeros 5 registros
@app.route('/show_first_5_records')
def show_records():
    records = get_first_5_records()
    response = {"message": "First 5 Records", "records": records}
    return jsonify(response)

@app.route('/recommendations')
def show_recommendations():
    # Hacer recomendaciones para el usuario 1
    userid = request.args

    id = userid.get("user_id")


    user_id_to_predict = id

    # Crear un DataFrame con una sola fila para el usuario a predecir
    single_user_df = spark.createDataFrame([(user_id_to_predict,)], ["userid"])

    # Hacer recomendaciones para el usuario
    recommendations_user_1 = model2.recommendForUserSubset(single_user_df, 10)

    # Obtener las recomendaciones
    recommendations = recommendations_user_1.collect()[0]['recommendations']

    # Formatear los resultados
    formatted_recommendations = [{"movieid": reco['movieid'], "rating": reco['rating']} for reco in recommendations]

    response = {"message": f"Top 10 Recommendations for User {user_id_to_predict}:", "recommendations": formatted_recommendations}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
