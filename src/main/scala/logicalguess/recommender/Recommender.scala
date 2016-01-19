package logicalguess.recommender

import org.apache.spark.mllib.recommendation.Rating


/**
 * Trait for Recommenders
 */
trait Recommender extends Serializable {

  class UserNotFoundException extends Exception

  /**
   * Recommendation of products based on previous ratings
   * @param ratings Previous ratings of the user
   * @return Ratings with recommended products
   */
  def recommendFromRatings(ratings: Seq[Rating], numberOfRecommendedProducts: Int = 10): Seq[Rating]

  /**
   * Recommendation of products based on previous ratings of given user
   * @param userId ID of the user
   * @return Ratings with recommended products
   */
  def recommendForUser(userId: Int, numberOfRecommendedProducts: Int = 10): Seq[Rating]
}
