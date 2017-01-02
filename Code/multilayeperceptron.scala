import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row

// Load the data stored in LIBSVM format as a DataFrame.
val data = MLUtils.loadLabeledData(sc, "/user/cloudera/Preprocessed_dataC.csv").toDF()
// Split the data into train and test
val splits = data.randomSplit(Array(0.9, 0.1), seed = 1234L)
val train = splits(0)
val test = splits(1)
// specify layers for the neural network:
// input layer of size 1, two intermediate of size 2 and 3
// and output of size 4
val layers = Array[Int](1, 2, 3, 4)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// train the model
val model = trainer.fit(train)
// compute precision on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
//Evaluation
val evaluator = new MulticlassClassificationEvaluator().setMetricName("precision")
val accuracy = evaluator.evaluate(result)
println("Accuracy = " + accuracy)
println("Precision:" + evaluator.evaluate(predictionAndLabels)*100)
