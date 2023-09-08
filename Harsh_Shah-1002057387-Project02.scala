// Databricks notebook source
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector, sum}

//Part 1
def computeGradientSummand(w: DenseVector[Double], dataPoint: LabeledPoint): DenseVector[Double] = {
  val x: DenseVector[Double] = DenseVector(dataPoint.features.toArray)
  val y: Double = dataPoint.label
  val prediction: Double = w.dot(x)
  val gradientSummand: DenseVector[Double] = (prediction - y) * x
  gradientSummand
}

val w = DenseVector(0.5, 0.3, 0.2) // Creating the weight vector w
val data = Seq(
  LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 3.0)), // First example: label=1.0, features=[1.0, 2.0, 3.0]
  LabeledPoint(2.0, Vectors.dense(4.0, 5.0, 6.0))  // Second example: label=2.0, features=[4.0, 5.0, 6.0]
)

val gradientSummandExample1 = computeGradientSummand(w, data(0)) 
val gradientSummandExample2 = computeGradientSummand(w, data(1))

println("Gradient Summand for Example 1:")
println(gradientSummandExample1)

println("Gradient Summand for Example 2:")
println(gradientSummandExample2)

//Part 2
def predict(w: DenseVector[Double], dataPoint: LabeledPoint): (Double, Double) = {
  val x: DenseVector[Double] = DenseVector(dataPoint.features.toArray)
  val y: Double = dataPoint.label
  val prediction: Double = w.dot(x)
  (y, prediction)
}

val dataRDD = spark.sparkContext.parallelize(data)

val predictionsRDD = dataRDD.map(dataPoint => predict(w, dataPoint))

predictionsRDD.collect().foreach { case (label, prediction) =>
  println(s"Label: $label, Prediction: $prediction")
}

//Part 3
def computeRMSE(predictionsRDD: org.apache.spark.rdd.RDD[(Double, Double)]): Double = {
  val n = predictionsRDD.count().toDouble
  val squaredErrorsSum = predictionsRDD.map { case (label, prediction) =>
    math.pow(label - prediction, 2)
  }.sum()
  math.sqrt(squaredErrorsSum / n)
}

val rmse = computeRMSE(predictionsRDD)
println(s"RMSE: $rmse")

//Part 4
// Gradient Descent Function
def gradientDescent(trainData: org.apache.spark.rdd.RDD[LabeledPoint]): (DenseVector[Double], Array[Double]) = {
  var w: DenseVector[Double] = DenseVector.zeros[Double](trainData.first().features.size)
  var alpha: Double = 0.01
  val numIterations: Int = 5
  val n: Double = trainData.count()

  val trainingErrors: Array[Double] = new Array[Double](numIterations)

  for (i <- 1 until ( numIterations + 1 )) {
    val gradientSum = trainData.map(dataPoint => computeGradientSummand(w, dataPoint)).reduce(_ + _)
    val deltaW = -alpha * (gradientSum) 

    w += deltaW
    alpha = alpha / ( n * math.sqrt(i)) // Update alpha using the formula provided
    println(s"After Iteration $i: alpha: $alpha")
    // Compute training error for current iteration
    val predictionsRDD = trainData.map(dataPoint => predict(w, dataPoint))
    val squaredErrorsSum = predictionsRDD.map { case (label, prediction) =>
      math.pow(label - prediction, 2)
    }.sum()
    trainingErrors(i - 1) = math.sqrt(squaredErrorsSum / n)
  }

  (w, trainingErrors)
}

val (weights, trainingErrors) = gradientDescent(dataRDD)

println(s"Weights: $weights")
println("Training Errors:")
trainingErrors.zipWithIndex.foreach { case (error, iteration) =>
  println(s"Iteration ${iteration + 1}: RMSE = $error")
}

// COMMAND ----------

//Bonus Part
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.{DenseMatrix => BreezeDenseMatrix, inv => breezeInv}

def ClosedFormSolution {

  val X: DenseMatrix = new DenseMatrix(3, 2, Array(1.0, 2.0, 3.0, 1.0, 2.0, 1.0))
  val y: DenseVector = new DenseVector(Array(3.0, 6.0, 5.0))

  // Calculate X^T * X
  val XT_X: DenseMatrix = X.transpose.multiply(X)

  // Calculate (X^T * X)^-1
  val XT_X_inverse: DenseMatrix = invertMatrix(XT_X)

  // Calculate X^T * y
  val XT_y: DenseVector = X.transpose.multiply(y)

  // Calculate the weight vector w: w = (X^T * X)^-1 * (X^T * y)
  val w: DenseVector = XT_X_inverse.multiply(XT_y)

  println("Weight vector w:")
  println(w)
}

// Function to invert a DenseMatrix
def invertMatrix(matrix: DenseMatrix): DenseMatrix = {
  val breezeMatrix = new BreezeDenseMatrix(matrix.numRows, matrix.numCols, matrix.toArray)
  val invMatrix = breezeInv(breezeMatrix)
  new DenseMatrix(matrix.numCols, matrix.numRows, invMatrix.toArray)
}

// Call the ClosedFormSolution method to execute the computation and print the result
ClosedFormSolution
