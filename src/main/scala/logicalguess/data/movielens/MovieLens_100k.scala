package logicalguess.data.movielens

import logicalguess.data.{DataProvider, WrongInputDataException}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD

/**
  * Created by logicalguess on 1/18/16.
  */
case class MovieLens_100k(dataDirectoryPath: String) extends DataProvider{
  override protected val ratings: RDD[Rating] = loadRatings(dataDirectoryPath)
  override protected val productNames: Map[Int, String] = loadProductNames(dataDirectoryPath)


  check()

  protected def loadRatings(dataDirectoryPath: String): RDD[Rating] = {

    /* Load the raw ratings data from a file */
    val rawData = env.Config.sc.textFile(dataDirectoryPath + "/u.data")

    /* Extract the user id, movie id and rating only from the dataset */
    val rawRatings = rawData.map(_.split("\t").take(3))

    /* Construct the RDD of Rating objects */
    val ratings = rawRatings.map {
      case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    ratings
  }

  protected def loadProductNames(dataDirectoryPath: String): Map[Int, String] = {
    /* Load item names to inspect the recommendations */
    val items = env.Config.sc.textFile(dataDirectoryPath + "/u.item")
    val pairRDD: RDD[(Int, String)] = items.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1)))
    val names = pairRDD.collect.toMap
    names
  }

  protected def check() = {
    val wrong = ratings.filter(r => (r.rating < 0 || (r.rating > 5) || (!productNames.contains(r.product))))
    if (wrong.count() != 0) throw new WrongInputDataException
  }
}
