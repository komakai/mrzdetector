package net.telepathix.mrzdetector

import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.utils.Converters
import java.lang.Integer.max
import java.lang.Integer.min
import java.lang.Math.cbrt
import java.lang.Math.pow
import java.lang.UnsupportedOperationException
import kotlin.math.*

val mrzDummy = run {
    System.loadLibrary("opencv_java4")
    System.loadLibrary("mrzdetector")
}

// Constants
const val maxCorners = 800
const val qualityLevel = 0.01
const val minDistance = 5.0
const val blockSize = 3
const val gradientSize = 3
const val lutInflectionPoint = 75
const val shortDimPartitions = 2
const val extremeCount = 180
const val extremeThreshold = -1080.0

// opencv2 extensions

fun Point.scale(factor: Double) {
    this.x *= factor
    this.y *= factor
}

operator fun Point.plus(rhs: Point) = Point(this.x + rhs.x, this.y + rhs.y)
operator fun Point.minus(rhs: Point) = Point(this.x - rhs.x, this.y - rhs.y)
operator fun Point.times(rhs: Double) = Point(this.x * rhs, this.y * rhs)
operator fun Point.div(rhs: Double) = Point(this.x / rhs, this.y / rhs)
fun Point.toMat(type: Int = CvType.CV_64F): Mat {
    val ret = Mat(2,1, type)
    ret.put(0, 0, this.x, this.y)
    return ret
}

fun ceil(size: Size) = Size(ceil(size.width), ceil(size.height))

val SIZE_3x3 = Size(3.0, 3.0)
val SIZE_3x2 = Size(3.0, 2.0)
val RECT_3x2 = Rect(0, 0, 3, 2)
val ID_3x3F:Mat = Mat.eye(SIZE_3x3, CvType.CV_32F)
val ID_3x2F:Mat = Mat.eye(SIZE_3x2, CvType.CV_32F)
val ID_3x3D:Mat = Mat.eye(SIZE_3x3, CvType.CV_64F)
val ID_3x2D:Mat = Mat.eye(SIZE_3x2, CvType.CV_64F)
val NULL_RECT = Rect(Int.MIN_VALUE, Int.MIN_VALUE,0,0)

typealias Vector = Point

fun Mat.safeSubmat(roi: Rect): Mat {
    val safeX = max(roi.x, 0); val safeY = max(roi.y, 0)
    return this.submat(Rect(safeX, safeY, min(roi.width, this.width() - safeX), min(roi.height, this.height() - safeY)))
}

operator fun Mat.plus(rhs: Mat): Mat {
    val dst = Mat()
    Core.add(this, rhs, dst)
    return dst
}

operator fun Mat.minus(rhs: Mat): Mat {
    val dst = Mat()
    Core.subtract(this, rhs, dst)
    return dst
}

operator fun Mat.div(rhs: Mat): Mat {
    val dst = Mat()
    Core.divide(this, rhs, dst)
    return dst
}

operator fun Mat.times(rhs: Point): Point {
    val res = this * rhs.toMat(this.type())
    return when {
        this.type() == CvType.CV_32F -> {
            val resBuffer = FloatArray(2)
            res.get(0, 0, resBuffer)
            Point(resBuffer[0].toDouble(), resBuffer[1].toDouble())
        }
        this.type() == CvType.CV_64F -> {
            val resBuffer = DoubleArray(2)
            res.get(0, 0, resBuffer)
            Point(resBuffer[0], resBuffer[1])
        }
        else -> {
            throw UnsupportedOperationException("Unsupported type:" + CvType.typeToString(this.type()))
        }
    }

}

@ExperimentalUnsignedTypes
class MrzDetector {

    fun densityFunc(x: Float) = (cbrt(x.toDouble())/2).toFloat()

    // adjust angle to be within range 0 to 180
    fun adjustAngle(angle: Float) = if (angle < 0) (angle + 180) else if (angle >= 180) angle - 180 else angle
    fun adjustAngle(angle: Int) = adjustAngle(angle.toFloat()).toInt()

    // adjust angle to be within range -90 to 90
    fun adjustAngle2(angle: Float) = if (angle < -90) angle + 180 else if (angle >= 90) angle - 180 else angle
    fun adjustAngle2(angle: Int) = adjustAngle2(angle.toFloat()).toInt()

    fun squareUp(dim1: Int, dim2: Int) = (dim1 + dim2 / (2 * shortDimPartitions)) / (dim2 / shortDimPartitions)

    fun partition(src: Mat): Pair<Int, Int> {
        val portrait = src.rows() > src.cols()
        val horizontalPartions = if (portrait) shortDimPartitions else squareUp(src.cols(), src.rows())
        val verticalPartions = if (portrait) squareUp(src.rows(), src.cols()) else shortDimPartitions
        return Pair(horizontalPartions, verticalPartions)
    }

    fun getPolar(point1: Point, point2: Point): Pair<Double, Float> {
        val yDiff = point2.y - point1.y
        val xDiff = point2.x - point1.x
        val angle = atan2(-yDiff, xDiff) * 180.0 / Math.PI
        return Pair(sqrt(yDiff * yDiff + xDiff * xDiff), adjustAngle(angle.toFloat()))
    }

    fun getTextOrientation(points: List<Point>, width: Double): Float {
        val dummy: List<MutableList<Triple<Int, Int, Double>>>? = null
        return getTextOrientation(points, width, dummy)
    }

    fun getTextOrientation(points: List<Point>, width: Double, pairs: List<MutableList<Triple<Int, Int, Double>>>?): Float {
        val angles = FloatArray(180)

        for (i in points.indices) {
            for (j in i + 1 until points.size) {
                val (distance, angle) = getPolar(points[i], points[j])
                if (distance < 0.7 * width) {
                    // give ourselves a score based on how close the points are
                    val score = ((((width - distance) * 8) / width) + 0.5).toFloat()
                    var mainAngle: Int
                    // for points close together the calculated angle is less precise
                    if (distance < 57) {
                        val baseAngle = adjustAngle(floor(angle + 0.5F) - 1F).toInt()
                        val densityOffset = (angle + 0.5F).rem(1F) - 1
                        val densityFuncVal1 = densityFunc(densityOffset)
                        val densityFuncVal2 = densityFunc(densityOffset + 1F)
                        angles[baseAngle] += score * (densityFuncVal1 + 0.5F)
                        angles[adjustAngle(baseAngle + 1)] += score * (densityFuncVal2 - densityFuncVal1)
                        angles[adjustAngle(baseAngle + 2)] += score * (0.5F - densityFuncVal2)
                        mainAngle = adjustAngle(baseAngle + 1)
                    } else {
                        val baseAngle = angle.toInt()
                        val densityFuncVal = angle.rem(1)
                        angles[baseAngle] += score * (1F - densityFuncVal)
                        angles[adjustAngle(baseAngle + 1)] += score * densityFuncVal
                        mainAngle = if (densityFuncVal <= 0.5) baseAngle else adjustAngle(baseAngle + 1)
                    }
                    pairs?.let {
                        it[mainAngle].add(Triple(i, j, distance))
                    }
                }
            }
        }

        val maxAngle: Int = angles.indices.maxByOrNull { angles[it] }!!
        val maxAngleFloat = maxAngle.toFloat()
        val maxAngleMinusOne = adjustAngle(maxAngle - 1)
        val maxAnglePlusOne = adjustAngle(maxAngle + 1)
        return adjustAngle2((angles[maxAngle] * maxAngleFloat + angles[maxAngleMinusOne] * (maxAngleFloat - 1) + angles[maxAnglePlusOne] * (maxAngleFloat + 1)) / (angles[maxAngleMinusOne] + angles[maxAngle] + angles[maxAnglePlusOne]))
    }

    fun brightnessCorrection(src: Mat): Mat {
        val lookUpTable = Mat(1, 256, CvType.CV_8U)
        val lookUpTableData = UByteArray((lookUpTable.total() * lookUpTable.channels()).toInt())
        for (i in 0..lutInflectionPoint) {
            val rangeLength = (lutInflectionPoint - 1).toDouble()
            lookUpTableData[i] = (rangeLength * pow(i.toDouble() / rangeLength, 2.0)).roundToInt().toUByte()
        }
        for (i in lutInflectionPoint..255) {
            val rangeLength = (255 - lutInflectionPoint).toFloat()
            lookUpTableData[i] = ((rangeLength * sqrt((i - lutInflectionPoint).toFloat() / rangeLength)).roundToInt() + lutInflectionPoint).toUByte()
        }
        lookUpTable.put(0, 0, lookUpTableData )
        val srcCorrected = Mat()
        Core.LUT(src, lookUpTable, srcCorrected)
        return srcCorrected
    }

    fun getGoodFeatures(src: Mat): List<Point> {
        val (horizontalPartions, verticalPartions) = partition(src)
        val corners = mutableListOf<Point>()
        for (i in 0 until horizontalPartions) {
            for (j in 0 until verticalPartions) {
                val partitionCornersMat = MatOfPoint()
                val topLeft = Point(((i * src.cols()) / horizontalPartions).toDouble(), (j * src.rows() / verticalPartions).toDouble())
                val bottomRight = Point((((i + 1) * src.cols()) / horizontalPartions).toDouble(), (((j + 1) * src.rows()) / verticalPartions).toDouble())
                Imgproc.goodFeaturesToTrack(src.submat(Rect(topLeft, bottomRight)), partitionCornersMat, maxCorners / (horizontalPartions * verticalPartions), qualityLevel, minDistance, Mat(), blockSize, gradientSize, false)
                val partitionCorners = mutableListOf<Point>()
                Converters.Mat_to_vector_Point(partitionCornersMat, partitionCorners)
                corners.addAll(partitionCorners.map { it + topLeft } )
            }
        }
        return corners
    }

    fun getWarpData(points: List<Point>, width: Double): Pair<Float, Point> {
        val pairs: List<MutableList<Triple<Int, Int, Double>>> = List(180) {
            mutableListOf()
        }
        val angle = getTextOrientation(points, width, pairs)
        val anglePairs = pairs[adjustAngle(angle).toInt()]
        val center = anglePairs.map { (points[it.first] + points[it.second]) / 2.0 }.fold(Point(0.0, 0.0)) { acc, point -> acc + point }
        center.scale(1.0/anglePairs.size.toDouble())
        return Pair(angle, center)
    }

    fun applyTransform(pointsMat: Mat, transform: Mat): Mat {
        val transformedPointsMat = Mat()
        if (transform.size() == SIZE_3x2) {
            Core.transform(pointsMat, transformedPointsMat, transform)
        } else {
            Core.perspectiveTransform(pointsMat, transformedPointsMat, transform)
        }
        return transformedPointsMat
    }

    fun transformPoints(points: List<Point>, transform: Mat): List<Point> {
        val pointsMat = Converters.vector_Point2f_to_Mat(points)
        val ret = mutableListOf<Point>()
        Converters.Mat_to_vector_Point2f(applyTransform(pointsMat, transform), ret)
        return ret
    }

    fun getAngleVector(angle: Float): Vector {
        val angleRadians = (angle * PI) / 180
        return Vector(cos(angleRadians), sin(angleRadians))
    }

    fun getSideSquared(p1: Point, p2: Point): Double {
        val xDiff = p2.x - p1.x
        val yDiff = p2.y - p1.y
        return xDiff * xDiff + yDiff * yDiff
    }

    fun getDewarpTransform(topWarpAngle: Float, topCenter: Point, bottomWarpAngle: Float, bottomCenter: Point): Triple<Mat, Double, Point> {
        val topGrad = getAngleVector(-topWarpAngle)
        val bottomGrad = getAngleVector(-bottomWarpAngle)
        val mat = ID_3x3F.clone()
        mat.submat(Rect(0, 0, 2,2)).put(0, 0, -topGrad.x, bottomGrad.x, -topGrad.y, bottomGrad.y)
        val invMat = mat.inv().submat(Rect(0, 0, 2, 2))
        val tempPoint = topCenter - bottomCenter
        val solution = invMat * tempPoint
        val vanishingPoint = Point(topCenter.x + topGrad.x * solution.x, topCenter.y + topGrad.y * solution.x)
        val VP2TC = sqrt(getSideSquared(vanishingPoint, topCenter))
        val VP2BC = sqrt(getSideSquared(vanishingPoint, bottomCenter))
        val vpDistance = (VP2TC + VP2BC) / 2
        val topRotationPoint = vanishingPoint + topGrad * vpDistance
        val bottomRotationPoint = vanishingPoint + bottomGrad * vpDistance
        val midAngle = (topWarpAngle + bottomWarpAngle).toDouble() / 2
        val midGrad = getAngleVector((-midAngle).toFloat())
        val topRotationBeforePoint1 = topRotationPoint + topGrad; val bottomRotationBeforePoint1 = bottomRotationPoint + bottomGrad
        val topRotationAfterPoint1 = topRotationPoint + midGrad; val bottomRotationAfterPoint1 = bottomRotationPoint + midGrad
        val topRotationBeforePoint2 = topRotationPoint - topGrad; val bottomRotationBeforePoint2 = bottomRotationPoint - bottomGrad
        val topRotationAfterPoint2 = topRotationPoint - midGrad; val bottomRotationAfterPoint2 = bottomRotationPoint - midGrad
        val beforePoints = MatOfPoint2f(topRotationPoint, topRotationBeforePoint1, topRotationBeforePoint2, bottomRotationPoint, bottomRotationBeforePoint1, bottomRotationBeforePoint2)
        val afterPoints = MatOfPoint2f(topRotationPoint, topRotationAfterPoint1, topRotationAfterPoint2, bottomRotationPoint, bottomRotationAfterPoint1, bottomRotationAfterPoint2)
        val homography = Calib3d.findHomography(beforePoints, afterPoints)
        return Triple(homography, midAngle, (topRotationPoint + bottomRotationPoint) / 2.0)
    }

    fun expand(m: Mat): Mat {
        if (m.size() == SIZE_3x3) {
            return m
        }
        val ret = Mat.eye(SIZE_3x3, m.type())
        m.copyTo(ret.submat(RECT_3x2))
        return ret
    }

    fun contract(m: Mat): Mat {
        return m.submat(RECT_3x2)
    }

    fun translate(mat: Mat, xShift: Double, yShift: Double) {
        mat.at<Double>(0,2).v += xShift
        mat.at<Double>(1,2).v += yShift
    }

    fun buildTransforms(transform: Mat, rotationAngle: Double, rotationCenter: Point, sizeIn: Size): Triple<Mat, Size, Mat> {
        val rotation = expand(Imgproc.getRotationMatrix2D(rotationCenter, rotationAngle, 1.0))
        val expandedTransform = expand(transform)
        val fullTransform = expandedTransform * rotation
        val vertices = listOf(Point(0.0, 0.0), Point(0.0, sizeIn.height), Point(sizeIn.width, sizeIn.height), Point(sizeIn.width, 0.0))
        val transformedVertices = transformPoints(vertices, fullTransform)
        val xVals = transformedVertices.map { it.x }
        val yVals = transformedVertices.map { it.y }
        val bbox = Rect2d(Point(xVals.minOrNull()!!, yVals.minOrNull()!!), Point(xVals.maxOrNull()!!, yVals.maxOrNull()!!))
        translate(fullTransform, -bbox.x, -bbox.y)

        val inverseTransform = fullTransform.inv()
        val rightSizedTransform = if (transform.size() == SIZE_3x3) fullTransform else contract(fullTransform)
        val rightSizedInverseTransform = if (transform.size() == SIZE_3x3) inverseTransform else contract(inverseTransform)
        return Triple(rightSizedTransform, ceil(bbox.size()), rightSizedInverseTransform)
    }

    fun transformMat(src: Mat, transform: Mat, bboxSize: Size, borderMode: Int = Core.BORDER_CONSTANT): Mat {
        val dst = Mat()
        if (transform.size() == SIZE_3x3) {
            Imgproc.warpPerspective(src, dst, transform, bboxSize, Imgproc.INTER_AREA, borderMode, Scalar(255.0, 255.0, 255.0, 255.0))
        } else {
            Imgproc.warpAffine(src, dst, transform, bboxSize, Imgproc.INTER_AREA, borderMode, Scalar(255.0, 255.0, 255.0, 255.0))
        }
        return dst
    }

    fun getArrowKernel(): Mat {
        val arrowKernel = Mat(5, 3, CvType.CV_32F)
        val twoOverRoot5 = 2F/sqrt(5F)
        val kernelData = floatArrayOf(0F, -twoOverRoot5 - 2, -1F, -twoOverRoot5, 0F, 0F, -1F, 4F, 3F + 4F * twoOverRoot5, -twoOverRoot5, 0F, 0F, 0F, -twoOverRoot5 - 2, -1F)
        arrowKernel.put(0, 0, kernelData)
        return arrowKernel
    }

    fun getStrokeWidthEstimate(src: Mat, extremes: List<Point>): Int {
        var count = 0; var total = 0
        for (extreme in extremes.subList(0, 40)) {
            val rightGap = (src.width() - extreme.x).toInt()
            val left = if (extreme.x.toInt() > 0) (extreme.x - 1).toInt() else 0
            val right = if (rightGap >= 25) extreme.x.toInt() + 25 else src.width() - 1
            val rightPixels = UByteArray(right - left)
            src.get(extreme.y.toInt(), extreme.x.toInt(), rightPixels)
            val initVal = rightPixels[0].toDouble()
            var foundLeft = false
            for ((index, pixel) in rightPixels.drop(1).withIndex()) {
                if (!foundLeft && pixel.toDouble() < initVal * 0.5) {
                    foundLeft = true
                    continue
                } else if (foundLeft && pixel.toDouble() > initVal * 0.7) {
                    count += 1
                    total += index
                    break
                }
            }
        }
        return if (count != 0) (total / count) else -1
    }

    fun getExtremePairs(src: Mat, extremes: List<Point>, rightPairScanBoxOffset: Int, rightPairScanBoxWidth: Int): List<Pair<Point, Point>> {
        val extremePairs = mutableListOf<Pair<Point, Point>>()
        outerExtreme@ for (extreme in extremes) {
            val extremeVal = src.at<Float>(extreme.y.toInt(), extreme.x.toInt()).v
            for (extreme2 in extremes) {
                if (extreme2 == extreme) {
                    break
                }
                if (abs(extreme.x - extreme2.x) <= 1 && abs(extreme.y - extreme2.y) <= 1) {
                    continue@outerExtreme
                }
            }
            if (extreme.y.toInt() == 0 || extreme.y.toInt() == src.height() - 1 || extreme.x.toInt() > src.width() - rightPairScanBoxOffset) {
                continue
            }
            val minMaxLoc = Core.minMaxLoc(src.safeSubmat(Rect(extreme.x.toInt() + rightPairScanBoxOffset, extreme.y.toInt() - 1, rightPairScanBoxWidth, 3)))
            if (minMaxLoc.maxVal > -extremeVal.toDouble() * 0.5) {
                extremePairs.add(Pair(extreme, Point(extreme.x + rightPairScanBoxOffset + minMaxLoc.maxLoc.x, extreme.y - 1 + minMaxLoc.maxLoc.y)))
            }
        }
        return extremePairs
    }

    fun getRanges(src: Mat, extremePairs: List<Pair<Point, Point>>): List<Pair<Int, Int>> {
        val ranges = mutableListOf<Pair<Int, Int>>()
        for (extremePair in extremePairs) {
            val extremeX = extremePair.first.x
            val extremeY = (extremePair.first.y + extremePair.second.y) / 2
            val width = 40
            val minMaxLocOuter = Core.minMaxLoc(src.safeSubmat(Rect(extremeX.toInt(), extremeY.toInt(), width, 1)))
            val whiteThreshold = (minMaxLocOuter.maxVal + minMaxLocOuter.minVal) / 2
            var yLo = 0; var yHi = src.height() - 1
            for (y in (0..extremeY.toInt()).reversed()) {
                val minMaxLoc = Core.minMaxLoc(src.safeSubmat(Rect(extremeX.toInt(), y, width, 1)))
                if (minMaxLoc.minVal > whiteThreshold) {
                    yLo = y + 1
                    break
                }
            }
            for (y in (extremeY.toInt() + 1)..src.height()) {
                val minMaxLoc = Core.minMaxLoc(src.safeSubmat(Rect(extremeX.toInt(), y, width, 1)))
                if (minMaxLoc.minVal > whiteThreshold) {
                    yHi = y - 1
                    break
                }
            }
            ranges.add(Pair(yLo, yHi))
        }
        ranges.sortByDescending { it.second - it.first }
        return ranges
    }

    fun mergeRanges(ranges: List<Pair<Int, Int>>): List<Triple<Int, Int, Int>> {
        val average = (ranges.map { it.second - it.first }.fold(0, Int::plus)).toDouble() / ranges.size.toDouble()
        val sortedFilteredRanges = ranges.dropWhile { (it.second - it.first) > average * 2.2 }

        val mergedRanges = mutableListOf<Triple<Int, Int, Int>>()
        outerRange@ for (range in sortedFilteredRanges) {
            for ((index, mergedRange) in mergedRanges.withIndex()) {
                if ((mergedRange.first <= range.first && range.first <= mergedRange.second) || (mergedRange.first <= range.second && range.second <= mergedRange.second)) {
                    mergedRanges[index] = Triple(min(mergedRange.first, range.first),max(mergedRange.second, range.second),mergedRanges[index].third + 1)
                    continue@outerRange
                }
            }
            mergedRanges.add(Triple(range.first, range.second, 1))
        }

        mergedRanges.sortBy { it.first }
        return mergedRanges
    }

    fun getRollupScore(range: IntRange, scores:List<Int>): Int {
        return scores.slice(range).fold(0, Int::plus)
    }

    fun findTextHEdge(start: Int, end: Int, minEnd: Int, mins: UByteArray, maxs: UByteArray, startThreshold: Double): Int {
        var threshold = startThreshold
        var currentRunLength = 0
        var runCount = 0
        var averageRun = 0.0
        val minMaxs = mutableListOf<Pair<UByte, UByte>>()
        var inDark = true
        var lastInDark = start
        var secondLastInDark = start

        for (x in if (start > end) (end..start).reversed() else (start..end)) {
            if (mins[x].toDouble() > threshold) {
                if (inDark) {
                    secondLastInDark = x - 1
                    inDark = false
                }
                currentRunLength += 1
                if (runCount > 0 && ((x > minEnd) == (end > start)) && currentRunLength.toDouble() > 2.5 * averageRun) {
                    return lastInDark
                }
            } else {
                if (!inDark) {
                    if (currentRunLength > 0) {
                        if (runCount > 0) {
                            threshold = minMaxs.map { it.first.toDouble() + it.second.toDouble() }.fold(0.0, Double::plus) / (2 * minMaxs.size).toDouble()
                            minMaxs.clear()
                        }
                        averageRun = ((currentRunLength + runCount).toDouble() * averageRun) / (runCount + 1).toDouble()
                        runCount += 1
                        currentRunLength = 0
                    }
                    inDark = true
                }
                lastInDark = x
                currentRunLength += 1
                if (runCount > 0 && ((x > minEnd) == (end > start)) && currentRunLength.toDouble() > 2.5 * averageRun) {
                    return secondLastInDark
                }
                minMaxs.add(Pair(mins[x], maxs[x]))
            }
        }
        return if (inDark) {
            secondLastInDark
        } else {
            lastInDark
        }
    }

    fun findTextVEdge(src: Mat, left: Int, right: Int, yEst: Int, isUp: Boolean): Int {
        val startMinMaxLoc = Core.minMaxLoc(src.safeSubmat(Rect(left, yEst + if (isUp) 1 else -1, right - left, 1)))
        val threshold = (startMinMaxLoc.minVal + startMinMaxLoc.maxVal * 4) / 5
        val start = yEst + (if (isUp) -1 else 1)
        val end = if (isUp) 0 else (src.height() - 1)

        for (y in if (start > end) (end..start).reversed() else (start..end)) {
            val dst = Mat()
            Core.reduce(src.safeSubmat(Rect(left, y, right - left, 1)), dst, 1, Core.REDUCE_AVG)
            val avg = dst.at<UByte>(0, 0).v
            if (avg.toDouble() > threshold) {
                return y
            }
        }
        return if (isUp) 0 else (src.height() - 1)
    }

    fun calcRowDims(src: Mat, excludeArea: List<Point>, points: List<Point>, yStartEst: Int, yEndEst: Int): Rect {
        val dstMinMat = Mat(); val dstMaxMat = Mat()
        val dstMin = UByteArray(src.width()); val dstMax = UByteArray(src.width())
        Core.reduce(src.submat(Rect(0, yStartEst, src.width(), yEndEst - yStartEst)), dstMinMat, 0, Core.REDUCE_MIN)
        dstMinMat.get(0, 0, dstMin)
        Core.reduce(src.submat(Rect(0, yStartEst, src.width(), yEndEst - yStartEst)), dstMaxMat, 0, Core.REDUCE_MAX)
        dstMaxMat.get(0, 0, dstMax)
        val xSortedPoints = points.sortedBy { it.x }
        val startLeftwardIndex = if (xSortedPoints.size > 1) 1 else 0
        val startLeftward = xSortedPoints[startLeftwardIndex]
        val minEndLeftward = xSortedPoints[max(startLeftwardIndex - 1, 0)]
        val startRightwardIndex = if (xSortedPoints.size > 1) (xSortedPoints.size - 2) else 0
        val startRightward = xSortedPoints[startRightwardIndex]
        val minEndRightward = xSortedPoints[min(startRightwardIndex + 1, xSortedPoints.count() - 1)]
        val startLeftMinMaxLoc = Core.minMaxLoc(src.safeSubmat(Rect(startLeftward.x.toInt() - 5, startLeftward.y.toInt() - 5, 10, 10)))
        val startLeftThreshold = (startLeftMinMaxLoc.minVal + startLeftMinMaxLoc.maxVal) / 2
        val rangeStartEst = getThresholds(excludeArea, yStartEst.toDouble())
        val rangeEndEst = getThresholds(excludeArea, yEndEst.toDouble())
        if (rangeStartEst == null || rangeEndEst == null) {
            return NULL_RECT
        }
        val left = findTextHEdge(startLeftward.x.toInt() - 1, max(rangeStartEst.start, rangeEndEst.start), minEndLeftward.x.toInt(), dstMin, dstMax, startLeftThreshold)
        val startRightMinMaxLoc = Core.minMaxLoc(src.safeSubmat(Rect(startRightward.x.toInt() - 5, startRightward.y.toInt() - 5, 10, 10)))
        val startRightThreshold = (startRightMinMaxLoc.minVal + startRightMinMaxLoc.maxVal) / 2
        val right = findTextHEdge(startRightward.x.toInt() + 1, min(rangeStartEst.end, rangeEndEst.end), minEndRightward.x.toInt(), dstMin, dstMax, startRightThreshold)
        if (left >= right) {
            return NULL_RECT
        }
        val top = findTextVEdge(src, left, right, yStartEst, true)
        val bottom = findTextVEdge(src, left, right, yEndEst, false)
        return Rect(left, top, (right - left), (bottom - top))
    }

    fun gapsOk(gaps: List<Double>, heights: List<Double>): Boolean {
        val gapsSorted = gaps.sorted().map { it + 10 }
        if (abs(gapsSorted[gaps.size - 1] - gapsSorted[0])/gapsSorted[gaps.size - 1] > 0.22) {
            return false
        }
        val heightsSorted = heights.sorted().map { it + 10 }
        if (abs(heightsSorted[heights.size - 1] - heightsSorted[0])/heightsSorted[gaps.size - 1] > 0.22) {
            return false
        }
        for (triple in (gaps zip (heights zip heights.drop(1)))) {
            if (triple.first > (triple.second.first + triple.second.second)) {
                return false
            }
        }
        return true
    }

    fun rangeOverlap(range1: Pair<Int, Int>, range2: Pair<Int, Int>): Pair<Int, Int> {
        if (range1.second <= range2.first || range2.second <= range1.first) {
            return Pair(0, 0)
        } else {
            return Pair(max(range1.first, range2.first), min(range1.second, range2.second))
        }
    }

    fun horizontalAlignmentOk(lefts: List<Int>, widths: List<Int>): Boolean {
        val ranges = (lefts zip widths).map { Pair(it.first, it.first + it.second) }
        val overlap = ranges.fold(Pair(Int.MIN_VALUE, Int.MAX_VALUE), ::rangeOverlap)
        val widest = widths.maxOrNull()!!.toDouble()
        return (overlap.second - overlap.first).toDouble() / widest > 0.78
    }

    fun findBestCandidate(rollupScoreMap: List<Pair<IntRange, Int>>, getRectFunc: (Int)->Rect): Pair<Pair<IntRange, Int>?, List<Rect>> {
        val rects = mutableMapOf<Int, Rect>()
        for (candidate in rollupScoreMap) {
            candidate.first.forEach { if (rects[it] == null) { rects[it] = getRectFunc(it) } }
            if (candidate.first.any { rects[it] == NULL_RECT }) {
                continue
            }
            val widths = candidate.first.map { rects[it]!!.width }
            val lefts = candidate.first.map { rects[it]!!.x }
            val heights = candidate.first.map { (rects[it]!!.height).toDouble() }
            val yStarts = candidate.first.map { rects[it]!!.y }.drop(1)
            val yEnds = candidate.first.map { rects[it]!!.y + rects[it]!!.height }.dropLast(1)
            val gaps = (yStarts zip yEnds).map { (it.first - it.second).toDouble() }
            if (candidate.first.count() == 3) {
                if (gapsOk(gaps, heights) && horizontalAlignmentOk(lefts, widths)) {
                    return Pair(candidate, candidate.first.map { rects[it]!! })
                }
            } else {
                if (gapsOk(gaps, heights) && horizontalAlignmentOk(lefts, widths)) {
                    return Pair(candidate, candidate.first.map { rects[it]!! })
                }
            }
        }
        return Pair(null, emptyList())
    }

    fun getMrz(src: Mat): Quadrilateral? {
        val srcGray = Mat()
        val width = src.width().toDouble()
        Imgproc.cvtColor(src, srcGray, Imgproc.COLOR_BGR2GRAY)

        val srcGrayCorrected = brightnessCorrection(srcGray)
        val goodFeatures = getGoodFeatures(srcGrayCorrected)

        val adjustedAngle = getTextOrientation(goodFeatures, width)

        val rotation = Imgproc.getRotationMatrix2D(Point(src.width().toDouble() / 2.0, src.height().toDouble() / 2.0), (-adjustedAngle).toDouble(), 1.0)

        val m1x = DoubleArray(3)
        rotation.get(1, 0, m1x)

        val point2RotatedY = goodFeatures.map {
            Pair(it, (m1x[0] * it.x) + (m1x[1] * it.y))
        }

        val (topWarpAngle, topCenter) = getWarpData(point2RotatedY.filter { it.second < src.height().toDouble() / 3.0 - m1x[2] }.map { it.first }, width)

        val (bottomWarpAngle, bottomCenter) = getWarpData(point2RotatedY.filter { it.second > (2.0 * src.height().toDouble() / 3.0) - m1x[2] }.map { it.first }, width)

        val angleDiff = adjustAngle(max(topWarpAngle, bottomWarpAngle) - min(topWarpAngle, bottomWarpAngle))
        val srcSize = src.size()

        val (dewarpMat, rotationAngle, rotationCenter) = if (angleDiff in 1.2..19.0)
                getDewarpTransform(topWarpAngle, topCenter, bottomWarpAngle, bottomCenter) else
                Triple(ID_3x2D.clone(), adjustedAngle.toDouble(), Point(srcSize.width / 2.0, srcSize.height / 2.0))
        val (transMat, bboxSize, untransMat) = buildTransforms(dewarpMat, -rotationAngle, rotationCenter, srcSize)

        val widthDouble = src.size().width; val heightDouble = src.size().height

        val transformedVertexPoints = transformPoints(listOf(Point(5.0, 5.0), Point(5.0, heightDouble - 5.0), Point(widthDouble - 5.0, heightDouble - 5.0), Point(widthDouble - 5.0, 5.0)), transMat)
        val transformedGoodFeaturesPoints = transformPoints(goodFeatures, transMat)
        val straightMat = transformMat(srcGrayCorrected, transMat, bboxSize)

        val arrowKernel = getArrowKernel()
        val arrowKernelResult = Mat()
        Imgproc.filter2D(straightMat, arrowKernelResult, CvType.CV_32F, arrowKernel, Point(1.0, 2.0))
        val extremes = getExtremes(arrowKernelResult, transformedVertexPoints, extremeThreshold, extremeCount)

        val strokeWidthEstimate = getStrokeWidthEstimate(straightMat, extremes)
        if (strokeWidthEstimate == -1) {
            return null
        }
        val rightPairScanBoxOffset = if (strokeWidthEstimate < 4) strokeWidthEstimate else (strokeWidthEstimate * 3) / 4
        val rightPairScanBoxWidth = if (strokeWidthEstimate < 8) 3 else (strokeWidthEstimate / 2)

        val extremePairs = getExtremePairs(arrowKernelResult, extremes, rightPairScanBoxOffset, rightPairScanBoxWidth)

        val ranges = getRanges(straightMat, extremePairs)
        val mergedRanges = mergeRanges(ranges)
        if (mergedRanges.size < 2) {
            return null
        }

        val scores = mergedRanges.map{ it.third }
        val candidateRanges = mergedRanges.indices.drop(2).map { (it-2..it) } + mergedRanges.indices.drop(1).map { (it-1..it) }
        val rollupScoreMap = candidateRanges.map { Pair(it, getRollupScore(it, scores)) }.sortedByDescending { it.second }

        val (bestCandidate, rects) = findBestCandidate(rollupScoreMap) { i ->
            calcRowDims(
                straightMat,
                transformedVertexPoints,
                extremePairs.map { (it.first + it.second) / 2.0 }
                    .filter { it.y >= mergedRanges[i].first && it.y <= mergedRanges[i].second },
                mergedRanges[i].first,
                mergedRanges[i].second
            )
        }

        if (bestCandidate == null) {
            print("MRZ not found")
            return null
        }

        val pointsInBestCandidate = transformedGoodFeaturesPoints.filter { point ->
            rects.map { Rect(it.x - 5, it.y - 5, it.width + 10, it.height + 10) }.any { it.contains(point) }
        }

        val adjustedTextAngle = getTextOrientation(pointsInBestCandidate, width)

        val xMin = rects.map { it.x }.minOrNull()!!; val xMax = rects.map { it.x + it.width }.maxOrNull()!!
        val yMin = rects.map { it.y }.minOrNull()!!; val yMax = rects.map { it.y + it.height }.maxOrNull()!!
        val xAdjust = (xMax - xMin) / 30.0; val yAdjust = ((yMax - yMin) / 10.0) + abs(sin(adjustedTextAngle * PI / 180.0)) * (xMax - xMin)/2.0
        val mrzCornersUntransformed = listOf(Point(xMin - xAdjust, yMin - yAdjust), Point(xMax + xAdjust, yMin - yAdjust), Point(xMax + xAdjust, yMax + yAdjust), Point(xMin - xAdjust, yMax + yAdjust))
        val mrzCorners = transformPoints(mrzCornersUntransformed, untransMat)
        val adjustMat = Imgproc.getRotationMatrix2D( (mrzCorners[0] + mrzCorners[2]) / 2.0, adjustedTextAngle.toDouble(), 1.0)
        val mrzCornersAdjusted = transformPoints(mrzCorners, adjustMat)
        return Quadrilateral(mrzCornersAdjusted[0], mrzCornersAdjusted[1], mrzCornersAdjusted[2], mrzCornersAdjusted[3])
    }
}