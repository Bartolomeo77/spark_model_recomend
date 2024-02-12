# spark_docker

# Spark Cluster with Docker & docker-compose

# General

Un clúster independiente de Spark simple para los fines de su entorno de pruebas. Una solución *docker-compose up* alejada de usted para su entorno de desarrollo Spark.

Docker Compose creará los siguientes contenedores:

container|Exposed ports
---|---
spark-master|9090 7077
spark-worker-1|9091
spark-worker-2|9092
demo-database|5432

# Installation

Los siguientes pasos le permitirán ejecutar los spark cluster's containers.

## Pre requisites

* Docker installed

* Docker compose  installed

## Build the image
```sh
cd spark_docker
```

```sh
docker build -t cluster-apache-spark:3.0.2 .
```

## Run the docker-compose

El último paso para crear su clúster de prueba será ejecutar compose file:

```sh
docker-compose up -d
```

## Validate your cluster

valide su clúster accediendo a la interfaz de usuario de Spark en cada URL maestra y de trabajador.

### Spark Master

http://localhost:9090/

### Spark Worker 1

http://localhost:9091/

### Spark Worker 2

http://localhost:9092/


# Resource Allocation 

Este clúster se envía con 2 trabajadores y un maestro, cada uno de los cuales tiene un conjunto particular de asignación de recursos (básicamente asignación de núcleos de RAM y CPU).
* La asignación predeterminada de núcleos de CPU para cada Spark Worker es 1 núcleo.

* La RAM predeterminada para cada spark-worker es 1024 MB.

* La asignación de RAM predeterminada para los ejecutores Spark es 256 MB.

* La asignación de RAM predeterminada para el controlador Spark es 128 MB

* Si desea modificar estas asignaciones, simplemente edite el archivo env/spark-worker.sh. o el archivo docker compose file

# Binded Volumes

Para facilitar la ejecución de la aplicación, he enviado dos montajes de volumen que se describen en el siguiente cuadro:

Host Mount|Container Mount|Purposse
---|---|---
apps|/opt/spark-apps| Se utiliza para que los archivos jar, py de su aplicación estén disponibles para todos los trabajadores y maestros.
data|/opt/spark-data| Se utiliza para que los datos de su aplicación estén disponibles para todos los trabajadores y maestros


# Run Sample applications

## NY Bus Stops Data [Pyspark]

Este programa simplemente carga datos archivados de [Movilens](https://grouplens.org/datasets/movielens/) y aplicar filtros básicos usando spark sql, el resultado se conserva en una tabla de postgresql.

La tabla cargada contendrá la siguiente estructura:

userid|movieid|ratingid|stamtimeid
---|---|---|---
196|242|3|881250949
			
Primero creamos la base de datos en postgres
Postgres:
```sh
docker exec -it spark_docker_demo-database_1 /bin/bash
```
```sh
postgres=# psql -U postgres
postgres=# CREATE DATABASE movilens;
postgres-# \c movilens
postgres-# SELECT * FROM movilens;
```


Para enviar la aplicación, conéctese a uno de los trabajadores o al maestro y ejecute:

```sh
docker exec -it spark_docker_spark-master_1 /bin/bash
root@959b7c958f23:/opt/spark#
```

```sh
/opt/spark/bin/spark-submit --master spark://spark-master:7077 \
--jars /opt/spark-apps/postgresql-42.2.22.jar \
--driver-memory 1G \
--executor-memory 1G \
/opt/spark-apps/subida20.py
```
Verificamos que los datos estan ahi es postgres

```sh
docker exec -it spark_docker_demo-database_1 /bin/bash
```
```sh
postgres=# psql -U postgres
postgres-# \c movilens
postgres-# SELECT * FROM movilens;
```

## FALTA MAS LA DOCUMENTACION ESTO ES PRUEBAS
## FALTA MAS LA DOCUMENTACION ESTO ES PRUEBAS
## FALTA MAS LA DOCUMENTACION ESTO ES PRUEBAS
## FALTA MAS LA DOCUMENTACION ESTO ES PRUEBAS
## FALTA MAS LA DOCUMENTACION ESTO ES PRUEBAS
## FALTA MAS LA DOCUMENTACION ESTO ES PRUEBAS
## FALTA MAS LA DOCUMENTACION ESTO ES PRUEBAS

## MTA Bus Analytics[Scala]

Este programa toma los datos archivados de [Movilens](https://grouplens.org/datasets/movielens/) y realizar algunas agregaciones, los resultados calculados se conservan en las tablas de PostgreSQL.

Cada tabla persistente corresponde a una agregación particular:

Table|Aggregation
---|---
day_summary|A summary of vehicles reporting, stops visited, average speed and distance traveled(all vehicles)
speed_excesses|Speed excesses calculated in a 5 minute window
average_speed|Average speed by vehicle
distance_traveled|Total Distance traveled by vehicle


To submit the app connect to one of the workers or the master and execute:

```sh
/opt/spark/bin/spark-submit --deploy-mode cluster \
--master spark://spark-master:7077 \
--total-executor-cores 1 \
--class mta.processing.MTAStatisticsApp \
--driver-memory 1G \
--executor-memory 1G \
--jars /opt/spark-apps/postgresql-42.2.22.jar \
--conf spark.driver.extraJavaOptions='-Dconfig-path=/opt/spark-apps/mta.conf' \
--conf spark.executor.extraJavaOptions='-Dconfig-path=/opt/spark-apps/mta.conf' \
/opt/spark-apps/mta-processing.jar
```

You will notice on the spark-ui a driver program and executor program running(In scala we can use deploy-mode cluster)

![alt text](./articles/images/stats-app.png "Spark UI with scala program running")


# Resumen

* Compilamos la imagen de Docker necesaria para ejecutar los contenedores Spark Master y Worker.

* Creamos un clúster independiente de Spark usando 2 nodos trabajadores y 1 nodo maestro usando Docker && Docker-compose.

* Copié los recursos necesarios para ejecutar aplicaciones de demostración.

* Ejecutamos una aplicación distribuida en casa (solo necesitamos suficientes núcleos de CPU y RAM para hacerlo).

# ¿Por qué un clúster independiente?

* Esto está diseñado para usarse con fines de prueba, básicamente una forma de ejecutar aplicaciones Spark distribuidas en su computadora portátil o de escritorio.

* Esto será útil para usar canalizaciones de CI/CD para sus aplicaciones Spark (un tema realmente difícil y candente)

# Pasos para conectar y usar un shell pyspark de forma interactiva

* Siga los pasos para ejecutar el archivo docker-compose. Puede reducir esto si es necesario a 1 trabajador.

```sh
docker-compose up --scale spark-worker=1
docker exec -it docker-spark-cluster_spark-worker_1 bash
apt update
apt install python3-pip
pip3 install pyspark
pyspark
```

# ¿Qué queda por hacer?

* En este momento, para ejecutar aplicaciones en el clúster en modo de implementación, es necesario especificar un puerto de controlador arbitrario.

* La entrada de envío de Spark en start-spark.sh no está implementada; el envío utilizado en las demostraciones puede activarse desde cualquier trabajador.
