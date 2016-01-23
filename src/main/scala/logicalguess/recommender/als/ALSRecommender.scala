package logicalguess.recommender.als

import logicalguess.data.DataProvider
import logicalguess.recommender.Recommender
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

/**
 * Recommender using Alternating Least Squares (ALS) from Spark's MLlib
 * @param rank Rank of the feature matrices (number of features).
 * @param lambda The regularization parameter
 * @param numIterations The number of iterations to run
 */
case class ALSRecommender(dataProvider: DataProvider, rank: Int, lambda: Double, numIterations: Int) extends Recommender {

  val sc = env.Config.sc
  val items = dataProvider.getProductNames()
  val ratings = dataProvider.getRatings()
  val ratingsGroupedByUser = ratings.map(rating => (rating.user, rating)).groupByKey().persist()
  val model = train()

  protected def train(userRatingsForRecommendationRDD: Option[RDD[Rating]] = None): MatrixFactorizationModel = {
    val trainingData = userRatingsForRecommendationRDD match {
      case Some(rdd) => ratings.union(rdd)
      case None => ratings
    }
    trainingData.persist

    ALS.train(trainingData, rank, numIterations, lambda)
  }

  override def recommendFromRatings(userRatingsForRecommendation: Seq[Rating], numberOfRecommendedProducts: Int) = {

    val model = train(Some(sc.parallelize(userRatingsForRecommendation)))
    val myRatedItemIds = userRatingsForRecommendation.map(_.product).toSet
    val candidates = sc.parallelize(items.keys.filter(!myRatedItemIds.contains(_)).toSeq)
    val recommendations = model
      .predict(candidates.map((0, _)))
      .collect
      .sortBy(-_.rating)
      .take(numberOfRecommendedProducts)
    recommendations
  }


  override def recommendForUser(userId: Int, numberOfRecommendedProducts: Int) = {
    val userRatings = {
      val ratings = ratingsGroupedByUser.lookup(userId)
      if (ratings.length <= 0) {
        throw new UserNotFoundException
      }
      ratings(0)
    }
    val ratedProducts = userRatings.map(rating => rating.product).toList

    val candidates = sc.parallelize(items.keys.filter(!ratedProducts.contains(_)).toSeq)

    val recommendations = model
      .predict(candidates.map((userId, _)))
      .collect
      .sortBy(-_.rating)
      .take(numberOfRecommendedProducts)
    recommendations
  }
}