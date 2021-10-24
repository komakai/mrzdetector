package net.telepathix.mrzdetector

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Range
import org.opencv.utils.Converters
import kotlin.math.ceil
import kotlin.math.floor

fun getThresholds(corners: List<Point>, y: Double): Range? {
    val thresholds = mutableListOf<Double>()
    for (cornerIndex in corners.indices) {
        val corner1 = corners[cornerIndex]; val corner2 = corners[(cornerIndex + 1) % corners.size]
        val yValues = listOf(y, corner1.y, corner2.y)
        val yValuesSorted = yValues.sorted()
        if (yValuesSorted[1] == y) {
            if (corner1.y == corner2.y) {
                continue;
            }
            val grad = (corner2.x - corner1.x) / (corner2.y - corner1.y);
            val intersectionX = corner1.x + (y - corner1.y) * grad;
            thresholds.add(intersectionX);
        }
    }
    thresholds.sort()
    return Range(floor(thresholds.first()).toInt(), ceil(thresholds.last()).toInt())
}

fun getExtremes(mat: Mat, excludeArea: List<Point>, threshold: Double, minPointCount: Int): List<Point> {
    val matExcludeArea = Converters.vector_Point2d_to_Mat(excludeArea)
    val matExtremes = nGetExtremes(mat.nativeObj, matExcludeArea.nativeObj, threshold, minPointCount)
    val ret = mutableListOf<Point>()
    Converters.Mat_to_vector_Point(Mat(matExtremes), ret)
    return ret
}

external fun nGetExtremes(mat: Long, excludeAreaCorners: Long, threshold: Double, minPointCount: Int): Long