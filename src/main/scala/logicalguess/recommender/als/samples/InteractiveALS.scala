package logicalguess.recommender.als.samples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}

// http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib.html

object InteractiveALS {

  def main(args: Array[String]) {

    println("\nStart running...")
    // remove verbose log
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("InteractiveALS")
      .set("spark.executor.memory", "512m")
    val sc = new SparkContext(conf)

    // load ratings and movie titles
     println("\nStep 1, load ratings and movie titles.")

    val movieLensHomeDir = "src/main/resources/ml-1m"

    val ratings = sc.textFile(movieLensHomeDir + "/ratings.dat").map { line =>
      val fields = line.split("::")
      // UserID::MovieID::Rating::Timestamp
      // e.g.
      // 1::1193::5::978300760
      // The RDD contains (Int, Rating) pairs.
      // We only keep the last digit of the timestamp as a random key: = fields(3).toLong % 10
      // The Rating class is a wrapper around tuple (user: Int, product: Int, rating: Double) 
      //      defined in org.apache.spark.mllib.recommendation package.
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      //
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }

    val movies = sc.textFile(movieLensHomeDir + "/movies.dat").map { line =>
      val fields = line.split("::")
      // MovieID::Title::Genres
      // e.g.
      // 1::Toy Story (1995)::Animation|Children's|Comedy
      //  read in movie ids and titles only
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect.toMap

    val numRatings = ratings.count
    // _._2 is the RDD ratings's Rating in the (Int, Rating) pairs
    // The Rating class is a wrapper around tuple (user: Int, product: Int, rating: Double)
    val numUsers = ratings.map(_._2.user).distinct.count
    val numMovies = ratings.map(_._2.product).distinct.count

    println("Loaded data: " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

    // We will use MLlib’s ALS to train a MatrixFactorizationModel, 
    // which takes a RDD[Rating] object as input. 
    // ALS has training parameters such as rank for matrix factors and regularization constants. 
    // To determine a good combination of the training parameters,
    // we split ratings into train (60%), validation (20%), and test (20%) based on the 
    // last digit of the timestamp, and cache them

    val numPartitions = 20
    // ratings format // format: (timestamp % 10, Rating(userId, movieId, rating))
    // The training set is 60%, it is based on the last digit of the timestamp
    // change to 30%, 10% and 10%
    val training = ratings.filter(x => x._1 <= 3)
                          .values
                          .repartition(numPartitions)
                          .persist
    // val validation = ratings.filter(x => x._1 >= 3 && x._1 < 8)
    val validation = ratings.filter(x => x._1 == 4 )
                            .values
                            .repartition(numPartitions)
                            .persist
    // val test = ratings.filter(x => x._1 >= 8).values.persist
    val test = ratings.filter(x => x._1 == 5).values.persist

    val numTraining = training.count
    val numValidation = validation.count
    val numTest = test.count

    println("\nStep 2, train with " + numTraining + " ratings.")
    // println("\nTraining: " + numTraining + " ratings, validation: " + numValidation + " ratings, test: " + numTest + " ratings.")

    // train models and evaluate them on the validation set
    // we will test only 8 combinations resulting from the cross product of 2 different ranks (8 and 12)
    // use rank 12 to reduce the running time
    // val ranks = List(8, 12)
    val ranks = List(12)

    // 2 different lambdas (1.0 and 10.0)
    val lambdas = List(0.1, 10.0)

    // two different numbers of iterations (10 and 20)
    // use numIters 20 to reduce the running time
    // val numIters = List(10, 20)
    val numIters = List(10)

    // We use the provided method computeRmse to compute the RMSE on the validation set for each model.
    // The model with the smallest RMSE on the validation set becomes the one selected 
    // and its RMSE on the test set is used as the final metric
    // import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      // in object ALS
      // def train(ratings: RDD[Rating], rank: Int, iterations: Int, lambda: Double) : MatrixFactorizationModel
      val model = ALS.train(training, rank, numIter, lambda)

      // def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long)
      // return  math.sqrt, type is double
      // model is from training.
      val validationRmse = computeRmse(model, validation, numValidation)
      // println("RMSE (validation) = " + validationRmse + " for the model trained with rank = " 
      //    + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        // println("inside bestModel  " +  bestModel);
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    // evaluate the best model on the test set
    println("\nStep 3, evaluate the best model on the test set.")

    val testRmse = computeRmse(bestModel.get, test, numTest)

    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
       + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")

    // create a naive baseline and compare it with the best model

    // val meanRating = training.union(validation).map(_.rating).mean
    // val baselineRmse = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating))
    //                                 .reduce(_ + _) / numTest)
    // val improvement = (baselineRmse - testRmse) / baselineRmse * 100
    // println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    // make personalized recommendations
    // in class MatrixFactorizationModel
    // def predict(userProducts: RDD[(Int, Int)]): RDD[Rating] 


    println("\nStep 4, provide with real time recommendations from user input.")
   
    var userid : Int = -1;
    while (userid !=0 ) {
    print( "\nPlease input user id (0 to exit): ")
    userid = Console.readInt
    if (userid < 0 || userid > 100) {
            println("User id is out of range, please input a valid user id (1 to 100). ")
    }else if ( userid !=0 ) { 
    val candidates = sc.parallelize(movies.keys.toSeq)

    // println("bestModel.get " + bestModel.get.predict(candidates.map((0, _))).take(20).foreach(println))
    // println("bestModel " +  bestModel )
    // use candidates.map((0, _)) returns an empty set.
    // use candidates.map((1, _)) returns a recommend list set.
    // it means the user real time data must be in the training set.
    val recommendations = bestModel.get
                                   .predict(candidates.map((userid, _)))
                                   .collect
                                   .sortBy(- _.rating)
                                   .take(30)
    var i = 1
    println("------------------------------------------------------------------ " )
    println(" Movies recommended for you:")
    recommendations.foreach { r =>
      println("\t" + "%2d".format(i) + ": " + movies(r.product))
      i += 1
    }
    println("------------------------------------------------------------------ " )
    }

    }
    // clean up
    println("Exiting...");
    sc.stop();
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long) = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
                                           .join(data.map(x => ((x.user, x.product), x.rating)))
                                           .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

  /** Asl for ratings from command-line. */
  // For each of the selected movies, we will ask you to give a rating (1-5) or 0 if you have never watched this movie. 
  // The method eclicitateRatings returns your ratings, where you receive a special user id 0. 
  //
  def askForRatings(movies: Seq[(Int, String)]) = {
    val prompt = "\nPlease rate the following movie (1-5 (best), or 0 if not seen):"
    println(prompt)
    // flatMap, takes every elements to a map
    // ratings format: (timestamp % 10, Rating(userId, movieId, rating))
    val ratings = movies.flatMap { x =>
      var rating: Option[Rating] = None
      var valid = false
      while (!valid) {
        print("\t" + x._2 + ": ")
        try {
          val r = Console.readInt
          if (r < 0 || r > 5) {
            println(prompt)
          } else {
            valid = true
            if (r > 0) {
              // Class Some[A] represents existing values of type A.
              // The Rating class is a wrapper around tuple (user: Int, product: Int, rating: Double)
              // you receive a special user id 0
              rating = Some(Rating(0, x._1, r))
            }
          }
        } catch {
          case e: Exception => println(prompt)
        }
      }
      rating match {
        // An iterator is not a collection, but rather a way to access the elements of a collection one by one.
        case Some(r) => Iterator(r)
        case None => Iterator.empty
      }
    }
    if(ratings.isEmpty) {
      error("No rating provided!")
    } else {
      ratings
    }
  }
}
