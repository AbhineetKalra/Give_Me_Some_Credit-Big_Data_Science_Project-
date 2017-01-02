import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark._
//Create RDD
var  data = sc.textFile("Preprocessed_data.csv")
//Remove Header

var changeddata= data.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

// Split data into training (70%) and test (30%).
val splits = changeddata.randomSplit(Array(0.7, 0.3), seed = 11L)
val training = splits(0)
val test = splits(1)
val trainingparsedData = training.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}


//Multinomial NaiveBayes with smoothening parameter=1.0
val model = NaiveBayes.train(trainingparsedData, lambda = 1.0, modelType = "multinomial")


val testparsedData = test.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}


//Calculating Accuracy of the model
val predictionAndLabel = testparsedData.map(p => (model.predict(p.features), p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

// Save and load model
model.save(sc, "target/tmp/myNaiveBayesModel")
val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
println("Accuracy is="+ accuracy)