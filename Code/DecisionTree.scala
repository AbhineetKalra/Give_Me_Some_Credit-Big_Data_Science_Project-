
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini


val loaddata = sc.textFile("Preprocessed_data.csv")

//loading data

var data= loaddata.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

//droping header



val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))



//splitting test and traing


val trainingparsedData = trainingData.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}


//setting label of training

val maxDepth = 5
val model = DecisionTree.train(trainingparsedData, Classification, Gini, maxDepth)

//training model using gini index and the specified depth


val testparsedData = testData.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}


//setting label of test


val labelAndPreds = testparsedData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

//cross-validation

val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testparsedData.count()
println("Test Error = " + testErr)

//printing error rate

println("Learned classification tree model:\n" + model.toDebugString)

//printing the tree
