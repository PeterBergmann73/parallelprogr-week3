package kmeans

import scala.annotation.tailrec
import scala.collection._
import scala.util.Random
import org.scalameter._
import common._

class KMeans {

  def generatePoints(k: Int, num: Int): Seq[Point] = {
    val randx = new Random(1)
    val randy = new Random(3)
    val randz = new Random(5)
    (0 until num)
      .map({ i =>
        val x = ((i + 1) % k) * 1.0 / k + randx.nextDouble() * 0.5
        val y = ((i + 5) % k) * 1.0 / k + randy.nextDouble() * 0.5
        val z = ((i + 7) % k) * 1.0 / k + randz.nextDouble() * 0.5
        new Point(x, y, z)
      }).to[mutable.ArrayBuffer]
  }

  def initializeMeans(k: Int, points: Seq[Point]): Seq[Point] = {
    val rand = new Random(7)
    (0 until k).map(_ => points(rand.nextInt(points.length))).to[mutable.ArrayBuffer]
  }

  def findClosest(p: Point, means: GenSeq[Point]): Point = {
    assert(means.size > 0)
    var minDistance = p.squareDistance(means(0))
    var closest = means(0)
    var i = 1
    while (i < means.length) {
      val distance = p.squareDistance(means(i))
      if (distance < minDistance) {
        minDistance = distance
        closest = means(i)
      }
      i += 1
    }
    closest
  }


  // The method takes a generic sequence of points and a generic sequence of means.
  // It returns a generic map collection,
  // which maps each mean to the sequence of points in the corresponding cluster.
  def classify(points: GenSeq[Point], means: GenSeq[Point]): GenMap[Point, GenSeq[Point]] = {
    // we cannot remove duplicate means - we will change the order of the means
    // val unique = means.toSet.toSeq

    // go through each point and find the closest mean
    val mapped0: GenSeq[(Point, Point)] = points.map(p => p -> findClosest(p, means))

    // now group
    val mapped: GenMap[Point, GenSeq[Point]] = mapped0.groupBy(_._2).map {
      case (k, seq) =>
        val m: GenSeq[Point] = seq.map {
          case (v1, v2) => v2
        }

        k -> m
    }

    // make sure that all the means are in the GenMap, even if their sequences are empty.
    val remapped: GenMap[Point, GenSeq[Point]] = means.map {
      m =>
        val v: GenSeq[Point] = mapped.getOrElse(m, GenSeq[Point]())
        m -> v
    }.toMap

    remapped
  }


  def findAverage(oldMean: Point, points: GenSeq[Point]): Point = if (points.length == 0) oldMean else {
    var x = 0.0
    var y = 0.0
    var z = 0.0
    points.seq.foreach { p =>
      x += p.x
      y += p.y
      z += p.z
    }
    new Point(x / points.length, y / points.length, z / points.length)
  }


  def update(classified: GenMap[Point, GenSeq[Point]], oldMeans: GenSeq[Point]): GenSeq[Point] = {
    oldMeans.map {
      om =>
        val v: GenSeq[Point] = classified.getOrElse(om, GenSeq[Point]())
        val m: Point = findAverage(om, v)
        m
    }
  }


  def converged(eta: Double)(oldMeans: GenSeq[Point], newMeans: GenSeq[Point]): Boolean = {
    oldMeans.zip(newMeans).forall {
      case (om, nm) => om.squareDistance(nm) <= eta
    }
  }


  @tailrec
  final def kMeans(points: GenSeq[Point], means: GenSeq[Point], eta: Double): GenSeq[Point] = {
    val classified: GenMap[Point, GenSeq[Point]] = classify(points, means)
    val newMeans: GenSeq[Point] = update(classified, means)
    val converged0 = converged(eta)(means, newMeans)

    // the implementation has to be tail recursive
    if (!converged0) kMeans(points, newMeans, eta) else newMeans
  }
}

/** Describes one point in three-dimensional space.
  *
  * Note: deliberately uses reference equality.
  */
class Point(val x: Double, val y: Double, val z: Double) {
  private def square(v: Double): Double = v * v

  def squareDistance(that: Point): Double = {
    square(that.x - x) + square(that.y - y) + square(that.z - z)
  }

  private def round(v: Double): Double = (v * 100).toInt / 100.0

  override def toString = s"(${round(x)}, ${round(y)}, ${round(z)})"
}


object KMeansRunner {

  val standardConfig = config(
    Key.exec.minWarmupRuns -> 20,
    Key.exec.maxWarmupRuns -> 40,
    Key.exec.benchRuns -> 25,
    Key.verbose -> true
  ) withWarmer (new Warmer.Default)

  def main(args: Array[String]) {
    val kMeans = new KMeans()

    val numPoints = 500000
    val eta = 0.01
    val k = 32
    val points = kMeans.generatePoints(k, numPoints)
    val means = kMeans.initializeMeans(k, points)

    val seqtime = standardConfig measure {
      kMeans.kMeans(points, means, eta)
    }
    println(s"sequential time: $seqtime ms")

    val partime = standardConfig measure {
      val parPoints = points.par
      val parMeans = means.par
      kMeans.kMeans(parPoints, parMeans, eta)
    }
    println(s"parallel time: $partime ms")
    println(s"speedup: ${seqtime / partime}")
  }

}
