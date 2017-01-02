import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini


import sqlContext.implicits._

/* Loading Data */
val loaddata1 = sc.textFile("Preprocessed_data.csv")

/* Processing for the Model */
var data1= loaddata1.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

val tData = data1.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

val TData = tData.toDF()

// Indexing Features 

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel") .fit(TData)

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(TData)

// Splitting into test & training 
val Array(trainingData, testData) = tData.randomSplit(Array(0.7, 0.3))

// Initializing RF Model numTrees set to 10 
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Creating compuattion piprline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

//Running Pipeline on training data
val model = pipeline.fit(trainingData.toDF())

val predictions = model.transform(testData.toDF())

//Evaluating the Model
predictions.select("predictedLabel", "label", "features").show(5)


val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]

