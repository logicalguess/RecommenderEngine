package env

import logicalguess.data.movielens.MovieLens_1m
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object Config {

  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

  val conf = new SparkConf().setAppName("SparkRecommender").setMaster("local[2]").set("spark.executor.memory","1g")
  val sc = new SparkContext(conf)
  val dataProvider = MovieLens_1m("src/main/resources/ml-1m")


}