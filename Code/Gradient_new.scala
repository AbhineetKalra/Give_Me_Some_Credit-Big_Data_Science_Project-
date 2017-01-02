import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
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

// Load and parse the data file
val loaddata1 = sc.textFile("/user/cloudera/Preprocessed_data.csv")

// Drop header content fom data
var data1= loaddata1.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

// Set label field in data
val tData = data1.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

// Convert data to a DataFrame.
val TData = tData.toDF()

// Index labels, adding metadata to the label column. Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel") .fit(TData)

// Automatically identify categorical features, and index them. Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(TData)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = tData.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val GT = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, GT, labelConverter))

// Train model.  This also runs the indexers.
val model = pipeline.fit(trainingData.toDF())

// Make predictions.
val predictions = model.transform(testData.toDF())

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")

val GTModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification forest model:\n" + GTModel.toDebugString)

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
