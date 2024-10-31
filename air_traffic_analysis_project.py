
#########
# Análisis de Datos del Tráfico Aéreo en el Aeropuerto de San Francisco #
#########

# Importar PySpark y configurar la sesión
from pyspark.sql import SparkSession

#########
# Crear una sesión de PySpark #
#########
spark = SparkSession.builder \
    .appName("San Francisco Air Traffic Analysis") \
    .getOrCreate()

#########
# Cargar los datos en PySpark #
#########
# Ruta al archivo de datos
data_path = "/mnt/data/air_traffic_data.csv"

# Cargar los datos en un DataFrame de PySpark
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Mostrar las primeras filas de los datos
df.show(5)

#########
# Análisis Inicial y Estructuración #
#########
# Mostrar el esquema de los datos
df.printSchema()

# Describir las columnas numéricas
df.describe().show()

#########
# Limpieza de los Datos #
#########
# Eliminar registros duplicados
df_cleaned = df.dropDuplicates()

# Seleccionar las columnas más relevantes
columns_to_keep = ["Airline", "Destination", "Passenger Count", "Flight Type"]
df_cleaned = df_cleaned.select(columns_to_keep)

# Mostrar las primeras filas después de la limpieza
df_cleaned.show(5)

#########
# Análisis Descriptivo #
#########
# Calcular estadísticas descriptivas por aerolínea
df_cleaned.groupBy("Airline").agg({"Passenger Count": "mean"}).show()

# Calcular la desviación estándar de los pasajeros por destino
df_cleaned.groupBy("Destination").agg({"Passenger Count": "stddev"}).show()

#########
# Crear una Matriz de Correlación #
#########
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Seleccionar columnas para la matriz de correlación
vector_col = "features"
assembler = VectorAssembler(inputCols=["Passenger Count"], outputCol=vector_col)
df_vector = assembler.transform(df_cleaned)

# Calcular la correlación
correlation_matrix = Correlation.corr(df_vector, vector_col)
print("Matriz de correlación:", correlation_matrix.collect()[0]["pearson({})".format(vector_col)])

#########
# Preparar Datos para Visualización en Tableau y D3.js #
#########
# Guardar los datos limpios en un archivo CSV para su visualización en Tableau
output_path = "/mnt/data/cleaned_air_traffic_data.csv"
df_cleaned.write.csv(output_path, header=True)

#########
# Finalización del Proyecto #
#########
print("El análisis de datos ha finalizado. Los datos limpios están listos para su visualización.")
