package env

import logicalguess.recommender.als.ALSRecommender

/**
  * Created by logicalguess on 1/18/16.
  */
object Run extends App {

  val recommender = new ALSRecommender(12, 0.01, 10)

  val userId = 1
  val recommendations = recommender.recommendForUser(userId)

  println("Recommendations for user " + userId)
  recommendations.foreach(println)

}
