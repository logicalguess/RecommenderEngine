package env

import logicalguess.recommender.als.ALSRecommender
import logicalguess.recommender.mahout.MahoutRecommender

/**
  * Created by logicalguess on 1/18/16.
  */
object Run extends App {

  val dataProvider  = env.Config.dataProvider

  //val recommender = ALSRecommender(dataProvider, 12, 0.01, 10)
  val recommender = MahoutRecommender(dataProvider)

  val userId = 1
  val recommendations = recommender.recommendForUser(userId)

  println("Recommendations for user " + userId)
  val productNames = dataProvider.getProductNames()
  recommendations.foreach(r => println(productNames(r.product), r.rating))

  println("Evaluator results")
  println(recommender.evaluate())
}
