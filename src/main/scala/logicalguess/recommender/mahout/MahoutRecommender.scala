package logicalguess.recommender.mahout

import java.io.File

import logicalguess.recommender.Recommender
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
import org.apache.mahout.cf.taste.recommender.RecommendedItem
import org.apache.spark.mllib.recommendation.Rating

case class MahoutRecommender() extends Recommender {

  val dataModel = new FileDataModel(new File("src/main/resources/ml-1m" + "/ratings.csv"))
  val userSimilarity = new PearsonCorrelationSimilarity(dataModel)
  val neighborhood = new NearestNUserNeighborhood(25, userSimilarity, dataModel)
  val recommender = new GenericUserBasedRecommender(dataModel, neighborhood, userSimilarity)

  override def recommendFromRatings(userRatingsForRecommendation: Seq[Rating], numberOfRecommendedProducts: Int) = {
    Nil
  }


  override def recommendForUser(userId: Int, numberOfRecommendedProducts: Int) = {
    import scala.collection.JavaConverters._
    val items: List[RecommendedItem] = recommender.recommend(userId, numberOfRecommendedProducts).asScala.toList
    for (item <- items) yield Rating(userId, item.getItemID().toInt, item.getValue())
  }
}