package logicalguess.data.movielens

import logicalguess.data.{WrongInputDataException, DataProvider}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD

/**
  * Created by logicalguess on 1/18/16.
  */
case class MovieLens_1m(dataDirectoryPath: String) extends DataProvider{
  override protected val ratings: RDD[Rating] = loadRatings(dataDirectoryPath)
  override protected val productNames: Map[Int, String] = loadProductNames(dataDirectoryPath)


  check()

  protected def loadRatings(dataDirectoryPath: String): RDD[Rating] = {

    val ratings = env.Config.sc.textFile(dataDirectoryPath + "/ratings.dat").map { line =>
      val fields = line.split("::")
      // format: Rating(userID, movieID, rating)
      (Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }
    ratings
  }

  protected def loadProductNames(dataDirectoryPath: String): Map[Int, String] = {
    val movies = env.Config.sc.textFile(dataDirectoryPath + "/movies.dat").map { line =>
      val fields = line.split("::")
      // format: (movieID, movieName)
      (fields(0).toInt, fields(1))
    }.collect.toMap
    movies
  }

  protected def check() = {
    val wrong = ratings.filter(r => (r.rating < 0 || (r.rating > 5) || (!productNames.contains(r.product))))
    if (wrong.count() != 0) throw new WrongInputDataException
  }
}
