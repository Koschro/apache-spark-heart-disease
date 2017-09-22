//Scala Assignment
import scala.io.Source
val csv = Source.fromURL("http://mlr.cs.umass.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data").mkString.split("\\r?\\n")
val rdd = sc.parallelize(csv)
rdd.first
val head = rdd.take(10)
val line = head(5)

case class MatchData(scores: Array[Double], matched: Double)

def toDouble(s: String) = {
if ("?".equals(s)) Double.NaN else s.toDouble
}

//parse process definition of our String data
def parse(line: String) = {
val pieces = line.split(',')
val scores = pieces.slice(0,13).map(toDouble)
val matched = pieces(13).toDouble
MatchData(scores, matched)
}
//execution of the parsing process
val md = parse(line)
val parsed = rdd.map(line => parse(line))

//funtion definition to make the NaNs 0s 
def naz(d: Double) = if (Double.NaN.equals(d)) 0.0 else d

//funtion definition that converts the values that are exceeding the ranges
def convert(s: Double) = { 
s match {
case 3.0  => 0.0
case 4.0  => 1.0
case 6.0  => 2.0
case 7.0  => 3.0
case _ => s
}
}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

val data = parsed.map { md =>
val scores = Array(0,1,2,3,4,5,6,7,8,9,10,11,12).map(i => naz(convert(md.scores(i))))
val featureVector = Vectors.dense(scores)
val label = if (md.matched != 0.0) 1.0 else 0.0
LabeledPoint(label, featureVector)
}

//We split our data for testing purposes
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int](1->2,2->5,5->2,6->3,8->2,10->4,11->4,12->4)
val numTrees = 10
val featureSubsetStrategy = "auto"
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

import org.apache.spark.mllib.tree.RandomForest

//We train our model
val model = RandomForest.trainClassifier(trainingData,numClasses, categoricalFeaturesInfo,
numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

val labelAndPreds = testData.map { point =>
val prediction = model.predict(point.features)
(point.label, prediction)
}
//Testing process of our model predictions
val testErr = labelAndPreds. filter(r => r._1 != 
r._2).count.toDouble/testData.count()