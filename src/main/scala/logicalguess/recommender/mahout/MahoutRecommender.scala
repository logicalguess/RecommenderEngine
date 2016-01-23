package logicalguess.recommender.mahout

import java.io.File

import logicalguess.data.DataProvider
import logicalguess.recommender.Recommender
import org.apache.mahout.cf.taste.eval.RecommenderBuilder
import org.apache.mahout.cf.taste.impl.common.FastByIDMap
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator
import org.apache.mahout.cf.taste.impl.model.{GenericDataModel, GenericUserPreferenceArray, GenericPreference}
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
import org.apache.mahout.cf.taste.model.{DataModel, PreferenceArray}
import org.apache.mahout.cf.taste.recommender
import org.apache.mahout.cf.taste.recommender.RecommendedItem
import org.apache.spark.mllib.recommendation.Rating

class Builder extends RecommenderBuilder {
  override def buildRecommender(dataModel: DataModel) = {
    val userSimilarity = new PearsonCorrelationSimilarity(dataModel)
    val neighborhood = new NearestNUserNeighborhood(25, userSimilarity, dataModel)
    new GenericUserBasedRecommender(dataModel, neighborhood, userSimilarity)
  }
}

case class MahoutRecommender(dataProvider: DataProvider) extends Recommender {

  import scala.collection.JavaConverters._

  val sc = env.Config.sc
  val preferences = dataProvider.getRatings()
    .map(r => new GenericPreference(r.user, r.product, r.rating.toFloat))

  val preferencesGroupedByUser = preferences.map(pref => (pref.getUserID, pref)).groupByKey().collect()

  val fastMap = new FastByIDMap[PreferenceArray]()
  for ((userId, prefs) <- preferencesGroupedByUser) {
    val prefArray = new GenericUserPreferenceArray(prefs.size)
    for ((pref, idx) <- prefs.zipWithIndex) prefArray.set(idx, pref)
    fastMap.put(userId, prefArray)
  }

  //val dataModel = new FileDataModel(new File("src/main/resources/ml-1m" + "/ratings.csv"))
  val dataModel = new GenericDataModel(fastMap)
  val builder = new Builder()
  val recommender = builder.buildRecommender(dataModel)

  override def recommendFromRatings(userRatingsForRecommendation: Seq[Rating], numberOfRecommendedProducts: Int) = {
    Nil
  }


  override def recommendForUser(userId: Int, numberOfRecommendedProducts: Int) = {
    val items: List[RecommendedItem] = recommender.recommend(userId, numberOfRecommendedProducts).asScala.toList
    for (item <- items) yield Rating(userId, item.getItemID().toInt, item.getValue())
  }

  override def evaluate() = {
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    evaluator.evaluate(builder, null, dataModel, 0.9, 1.0)
  }
}