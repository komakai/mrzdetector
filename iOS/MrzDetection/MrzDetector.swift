//
//  MrzDetector.swift
//  MrzDetection
//
//  Created by Giles Payne on 2021/10/10.
//

import Foundation
import opencv2

// Constants
let maxCorners:Int32 = 800
let qualityLevel = 0.01
let minDistance = 5.0
let blockSize:Int32 = 3
let gradientSize:Int32 = 3
let lutInflectionPoint = 75
let shortDimPartitions:Int32 = 2
let extremeCount:Int32 = 180
let extremeThreshold: Float = -1080

// opencv2 extensions
extension Size {
    convenience init(_ size2f: Size2f) {
        self.init(width: Int32(ceil(size2f.width)), height: Int32(ceil(size2f.height)))
    }
}

extension Size2f {
    convenience init(_ size: Size) {
        self.init(width: Float(size.width), height: Float(size.height))
    }
}

extension Point {
    convenience init(_ point2f: Point2f) {
        self.init(x: Int32(point2f.x), y: Int32(point2f.y))
    }

    convenience init(_ point2d: Point2d) {
        self.init(x: Int32(point2d.x), y: Int32(point2d.y))
    }

    func scale(factor: Double) {
        self.x = Int32(round(Double(self.x) * factor))
        self.y = Int32(round(Double(self.y) * factor))
    }

    static func +(lhs: Point, rhs: Point) -> Point {
        Point(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }

    static func -(lhs: Point, rhs: Point) -> Point {
        Point(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
    }

    static func *(lhs: Point, rhs: Int32) -> Point {
        Point(x: lhs.x * rhs, y: lhs.y * rhs)
    }

    static func /(lhs: Point, rhs: Int32) -> Point {
        Point(x: lhs.x / rhs, y: lhs.y / rhs)
    }

    func toMat() -> Mat {
        let ret = Mat(rows: 2, cols: 1, type: CvType.CV_32S)
        try! ret.put(row: 0, col: 0, data: [self.x, self.y])
        return ret
    }
}

extension Point2f {
    convenience init(_ point: Point) {
        self.init(x: Float(point.x), y: Float(point.y))
    }
    
    convenience init(_ point2d: Point2d) {
        self.init(x: Float(point2d.x), y: Float(point2d.y))
    }

    func scale(factor: Double) {
        self.x = Float(Double(self.x) * factor)
        self.y = Float(Double(self.y) * factor)
    }

    static func +(lhs: Point2f, rhs: Point2f) -> Point2f {
        Point2f(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }
    
    static func -(lhs: Point2f, rhs: Point2f) -> Point2f {
        Point2f(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
    }

    static func *(lhs: Point2f, rhs: Float) -> Point2f {
        Point2f(x: lhs.x * rhs, y: lhs.y * rhs)
    }

    static func /(lhs: Point2f, rhs: Float) -> Point2f {
        Point2f(x: lhs.x / rhs, y: lhs.y / rhs)
    }

    func toMat() -> Mat {
        let ret = Mat(rows: 2, cols: 1, type: CvType.CV_32F)
        try! ret.put(row: 0, col: 0, data: [self.x, self.y])
        return ret
    }
}

extension Point2d {
    convenience init(_ point: Point) {
        self.init(x: Double(point.x), y: Double(point.y))
    }

    convenience init(_ point2f: Point2f) {
        self.init(x: Double(point2f.x), y: Double(point2f.y))
    }

    func scale(factor: Double) {
        self.x *= factor
        self.y *= factor
    }

    static func +(lhs: Point2d, rhs: Point2d) -> Point2d {
        Point2d(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }
    
    static func -(lhs: Point2d, rhs: Point2d) -> Point2d {
        Point2d(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
    }

    static func *(lhs: Point2d, rhs: Double) -> Point2d {
        Point2d(x: lhs.x * rhs, y: lhs.y * rhs)
    }

    static func /(lhs: Point2d, rhs: Double) -> Point2d {
        Point2d(x: lhs.x / rhs, y: lhs.y / rhs)
    }

    func toMat() -> Mat {
        let ret = Mat(rows: 2, cols: 1, type: CvType.CV_64F)
        try! ret.put(row: 0, col: 0, data: [self.x, self.y])
        return ret
    }
}

let SIZE_3x3 = Size(width: 3, height: 3)
let SIZE_3x2 = Size(width: 3, height: 2)
let RECT_3x2 = Rect(x: 0, y: 0, width: 3, height: 2)
let ID_3x3F = Mat.eye(size: SIZE_3x3, type: CvType.CV_32F)
let ID_3x2F = Mat.eye(size: SIZE_3x2, type: CvType.CV_32F)
let ID_3x3D = Mat.eye(size: SIZE_3x3, type: CvType.CV_64F)
let ID_3x2D = Mat.eye(size: SIZE_3x2, type: CvType.CV_64F)
let NULL_RECT = Rect(x: Int32.min, y: Int32.min, width: 0, height: 0)

typealias Vector = Point
typealias Vector2f = Point2f
typealias Vector2d = Point2d
typealias IntRange = Swift.ClosedRange<Int>


let OpenCVErrorDomain = "OpenCVErrorDomain"

enum OpenCVError : Int {
    case IncompatibleDataType = 10001
    case IncompatibleBufferSize
    case IncompatibleMatSize
}

func throwIncompatibleDataType(typeName: String) throws {
    throw NSError(
        domain: OpenCVErrorDomain,
        code: OpenCVError.IncompatibleDataType.rawValue,
        userInfo: [
            NSLocalizedDescriptionKey: "Incompatible Mat type \(typeName)"
        ]
    )
}

func throwIncompatibleBufferSize(count: Int, channels: Int32) throws {
    throw NSError(
        domain: OpenCVErrorDomain,
        code: OpenCVError.IncompatibleBufferSize.rawValue,
        userInfo: [
            NSLocalizedDescriptionKey: "Provided data element number \(count) should be multiple of the Mat channels count \(channels)"
        ]
    )
}

func throwIncompatibleMatSize(expectedRowCount: Int32, expectedColCount: Int32) throws {
    throw NSError(
        domain: OpenCVErrorDomain,
        code: OpenCVError.IncompatibleMatSize.rawValue,
        userInfo: [
            NSLocalizedDescriptionKey: "Provided Mat should be of size (rows: \(expectedRowCount) cols: \(expectedColCount))"
        ]
    )
}

extension Mat {
    func safeSubmat(roi: Rect) -> Mat {
        let safeX = max(roi.x, 0), safeY = max(roi.y, 0)
        return self.submat(roi: Rect(x: safeX, y: safeY, width: min(roi.width, self.width() - safeX), height: min(roi.height, self.height() - safeY)))
    }
    
    static func +(lhs: Mat, rhs: Mat) -> Mat {
        let dst = Mat()
        Core.add(src1: lhs, src2: rhs, dst: dst)
        return dst
    }
    
    static func -(lhs: Mat, rhs: Mat) -> Mat {
        let dst = Mat()
        Core.subtract(src1: lhs, src2: rhs, dst: dst)
        return dst
    }

    static func /(lhs: Mat, rhs: Mat) -> Mat {
        let dst = Mat()
        Core.divide(src1: lhs, src2: rhs, dst: dst)
        return dst
    }

    static func *(lhs: Mat, rhs: Point) throws -> Point {
        if lhs.type() != CvType.CV_32S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: lhs.type()))
        } else if lhs.rows() != 2 || lhs.cols() != 2 {
            try throwIncompatibleMatSize(expectedRowCount: 2, expectedColCount: 1)
        }
        let res = lhs * rhs.toMat()
        var resBuffer = [Int32](repeating: 0, count: 2)
        try res.get(row: 0, col: 0, data: &resBuffer)
        return Point(x: resBuffer[0], y: resBuffer[1])
    }

    static func *(lhs: Mat, rhs: Point2f) throws -> Point2f {
        if lhs.type() != CvType.CV_32F {
            try throwIncompatibleDataType(typeName: CvType.type(toString: lhs.type()))
        } else if lhs.rows() != 2 || lhs.cols() != 2 {
            try throwIncompatibleMatSize(expectedRowCount: 2, expectedColCount: 1)
        }
        let res = lhs * rhs.toMat()
        var resBuffer = [Float](repeating: 0, count: 2)
        try res.get(row: 0, col: 0, data: &resBuffer)
        return Point2f(x: resBuffer[0], y: resBuffer[1])
    }

    static func *(lhs: Mat, rhs: Point2d) throws -> Point2d {
        if lhs.type() != CvType.CV_64F {
            try throwIncompatibleDataType(typeName: CvType.type(toString: lhs.type()))
        } else if lhs.rows() != 2 || lhs.cols() != 2 {
            try throwIncompatibleMatSize(expectedRowCount: 2, expectedColCount: 1)
        }
        let res = lhs * rhs.toMat()
        var resBuffer = [Double](repeating: 0, count: 2)
        try res.get(row: 0, col: 0, data: &resBuffer)
        return Point2d(x: resBuffer[0], y: resBuffer[1])
    }
}


public class MrzDetector {
    
    func densityFunc(_ x: Float) -> Float {
        cbrt(x)/2
    }

    // adjust angle to be within range 0 to 180
    func adjustAngle(_ angle: Float) -> Float {
        angle < 0 ? angle + 180 : (angle >= 180 ? angle - 180 : angle)
    }

    func adjustAngle(_ angle: Int) -> Int {
        Int(adjustAngle(Float(angle)))
    }

    // adjust angle to be within range -90 to 90
    func adjustAngle2(_ angle: Float) -> Float {
        angle < -90 ? angle + 180 : (angle >= 90 ? angle - 180 : angle)
    }

    func adjustAngle2(_ angle: Int) -> Int {
        Int(adjustAngle2(Float(angle)))
    }

    func squareUp(_ dim1: Int32, _ dim2: Int32) -> Int32 {
        (dim1 + dim2 / (2 * shortDimPartitions)) / (dim2 / shortDimPartitions)
    }

    func partition(_ src: Mat) -> (Int32, Int32) {
        let portrait = src.rows() > src.cols()
        let horizontalPartions = portrait ? shortDimPartitions : squareUp(src.cols(), src.rows())
        let verticalPartions = portrait ? squareUp(src.rows(), src.cols()) : shortDimPartitions
        return (horizontalPartions, verticalPartions)
    }

    func getPolar(_ point1: Point, _ point2: Point) -> (Double, Float) {
        let yDiff = point2.y - point1.y
        let xDiff = point2.x - point1.x
        let angle = atan2(Float(-yDiff), Float(xDiff)) * 180.0 / Float.pi
        return (sqrt(Double(yDiff * yDiff + xDiff * xDiff)), adjustAngle(angle))
    }

    func getTextOrientation(points: [Point], width: Double) -> Float {
        var dummy:[[(Int, Int, Double)]]? = nil
        return getTextOrientation(points: points, width: width, pairs: &dummy)
    }

    func getTextOrientation(points: [Point], width: Double, pairs:inout [[(Int, Int, Double)]]?) -> Float {
        var angles = [Float](repeating: 0, count: 180)

        for i in points.startIndex..<points.endIndex {
            for j in i+1..<points.endIndex {
                let (distance, angle) = getPolar(points[i], points[j])
                if distance < 0.7 * width {
                    // give ourselves a score based on how close the points are
                    let score = Float((((width - distance) * 8) / width) + 0.5)
                    var mainAngle: Int
                    // for points close together the calculated angle is less precise
                    if distance < 57 {
                        let baseAngle = Int(adjustAngle(floor(angle + 0.5) - 1.0))
                        let densityOffset = modf(angle + 0.5).1 - 1
                        let densityFuncVal1 = densityFunc(densityOffset)
                        let densityFuncVal2 = densityFunc(densityOffset + 1.0)
                        angles[baseAngle] += score * (densityFuncVal1 + 0.5)
                        angles[adjustAngle(baseAngle + 1)] += score * (densityFuncVal2 - densityFuncVal1)
                        angles[adjustAngle(baseAngle + 2)] += score * (0.5 - densityFuncVal2)
                        mainAngle = adjustAngle(baseAngle + 1)
                    } else {
                        let baseAngle = Int(angle)
                        let densityFuncVal = modf(angle).1
                        angles[baseAngle] += score * (1.0 - densityFuncVal)
                        angles[adjustAngle(baseAngle + 1)] += score * densityFuncVal
                        mainAngle = densityFuncVal <= 0.5 ? baseAngle : adjustAngle(baseAngle + 1)
                    }
                    if pairs != nil {
                        pairs![mainAngle].append((i, j, distance))
                    }
                }
            }
        }

        let maxAngle: Int = angles.indices.max { angles[$0] < angles[$1] }!
        let maxAngleFloat = Float(maxAngle)
        let maxAngleMinusOne = adjustAngle(maxAngle - 1)
        let maxAnglePlusOne = adjustAngle(maxAngle + 1)
        return adjustAngle2((angles[maxAngle] * maxAngleFloat + angles[maxAngleMinusOne] * (maxAngleFloat - 1) + angles[maxAnglePlusOne] * (maxAngleFloat + 1)) / (angles[maxAngleMinusOne] + angles[maxAngle] + angles[maxAnglePlusOne]))
    }
    
    func brightnessCorrection(_ src: Mat) -> Mat {
        let lookUpTable = Mat(rows: 1, cols: 256, type: CvType.CV_8U)
        var lookUpTableData = [UInt8](repeating: 0, count: lookUpTable.total()*Int(lookUpTable.channels()))
        for i in 0..<lutInflectionPoint {
            let rangeLength = Float(lutInflectionPoint - 1)
            lookUpTableData[i] = UInt8(round(rangeLength * pow(Float(i)/rangeLength, 2)))
        }
        for i in lutInflectionPoint..<256 {
            let rangeLength = Float(255 - lutInflectionPoint)
            lookUpTableData[i] = UInt8(round(rangeLength * sqrt(Float(i - lutInflectionPoint)/rangeLength))) + UInt8(lutInflectionPoint)
        }
        try! lookUpTable.put(row: 0, col: 0, data: lookUpTableData )
        let srcCorrected = Mat()
        Core.LUT(src: src, lut: lookUpTable, dst: srcCorrected)
        return srcCorrected
    }

    func getGoodFeatures(_ src: Mat) -> [Point] {
        let (horizontalPartions, verticalPartions) = partition(src)
        var corners = [Point]()
        for i in 0..<horizontalPartions {
            for j in 0..<verticalPartions {
                var partionCorners = [Point]()
                let topLeft = Point(x: (i * src.cols()) / horizontalPartions, y: j * src.rows() / verticalPartions)
                let bottomRight = Point(x: ((i + 1) * src.cols()) / horizontalPartions , y: ((j + 1) * src.rows()) / verticalPartions)

                Imgproc.goodFeaturesToTrack(image: src.submat(roi: Rect(point: topLeft, point: bottomRight)), corners: &partionCorners, maxCorners: maxCorners / (horizontalPartions * verticalPartions), qualityLevel: qualityLevel, minDistance: minDistance, mask: Mat(), blockSize: blockSize, gradientSize: gradientSize, useHarrisDetector: false)
                corners.append(contentsOf: partionCorners.map { $0 + topLeft })
            }
        }
        return corners
    }
 
    func getWarpData(points: [Point], width: Double) -> (Float, Point) {
        var pairs: [[(Int, Int, Double)]]? = [[(Int, Int, Double)]](repeating: [(Int, Int, Double)](), count: 180)
        let angle = getTextOrientation(points:points, width: width, pairs: &pairs)
        let anglePairs = pairs![Int(adjustAngle(angle))]
        let center = anglePairs.map { (points[$0.0] + points[$0.1]) / 2 }.reduce(Point(x: 0, y: 0)) { $0 + $1 }
        center.scale(factor: 1/Double(anglePairs.count))
        return (angle, center)
    }
    
    func applyTransform(pointsMat: Mat, transform: Mat) -> Mat {
        let transformedPointsMat = Mat()
        if transform.size() == SIZE_3x2 {
            Core.transform(src: pointsMat, dst: transformedPointsMat, m: transform)
        } else {
            Core.perspectiveTransform(src: pointsMat, dst: transformedPointsMat, m: transform)
        }
        return transformedPointsMat
    }

    func transformPoints(points: [Point2f], transform: Mat) -> [Point2f] {
        let pointsMat = Converters.vector_Point2f_to_Mat(points)
        return Converters.Mat_to_vector_Point2f(applyTransform(pointsMat: pointsMat, transform: transform))
    }

    func transformPoints(points: [Point2d], transform: Mat) -> [Point2d] {
        let pointsMat = Converters.vector_Point2d_to_Mat(points)
        return Converters.Mat_to_vector_Point2d(applyTransform(pointsMat: pointsMat, transform: transform))
    }

    func getAngleVector(angle: Float) -> Vector2f {
        let angleRadians = (angle * Float.pi) / 180
        return Vector2f(x: cos(angleRadians), y: sin(angleRadians))
    }

    func getSideSquared(p1: Point2f, p2: Point2f) -> Float {
        let xDiff = p2.x - p1.x
        let yDiff = p2.y - p1.y
        return xDiff * xDiff + yDiff * yDiff
    }

    func getDewarpTransform(topWarpAngle: Float, topCenter: Point, bottomWarpAngle: Float, bottomCenter: Point) -> (Mat, Double, Point2f) {
        let topGrad = getAngleVector(angle: -topWarpAngle)
        let bottomGrad = getAngleVector(angle: -bottomWarpAngle)
        let mat = ID_3x3F.clone()
        try! mat.submat(roi: Rect(x: 0, y: 0, width: 2, height: 2)).put(row: 0, col: 0, data: [-topGrad.x, bottomGrad.x, -topGrad.y, bottomGrad.y])
        let invMat = mat.inv().submat(roi: Rect(x: 0, y: 0, width: 2, height: 2))
        let topCenterF = Point2f(topCenter), bottomCenterF = Point2f(bottomCenter)
        let tempPoint = topCenterF - bottomCenterF
        let solution = try! invMat * tempPoint
        let vanishingPoint = Point2f(x: topCenterF.x + topGrad.x * solution.x, y: topCenterF.y + topGrad.y * solution.x)
        let VP2TC = sqrt(getSideSquared(p1: vanishingPoint, p2: topCenterF))
        let VP2BC = sqrt(getSideSquared(p1: vanishingPoint, p2: bottomCenterF))
        let vpDistance = (VP2TC + VP2BC) / 2
        let topRotationPoint = vanishingPoint + topGrad * vpDistance
        let bottomRotationPoint = vanishingPoint + bottomGrad * vpDistance
        let midAngle = Double(topWarpAngle + bottomWarpAngle) / 2
        let midGrad = getAngleVector(angle: Float(-midAngle))
        let topRotationBeforePoint1 = topRotationPoint + topGrad, bottomRotationBeforePoint1 = bottomRotationPoint + bottomGrad
        let topRotationAfterPoint1 = topRotationPoint + midGrad, bottomRotationAfterPoint1 = bottomRotationPoint + midGrad
        let topRotationBeforePoint2 = topRotationPoint - topGrad, bottomRotationBeforePoint2 = bottomRotationPoint - bottomGrad
        let topRotationAfterPoint2 = topRotationPoint - midGrad, bottomRotationAfterPoint2 = bottomRotationPoint - midGrad
        let beforePoints = [topRotationPoint, topRotationBeforePoint1, topRotationBeforePoint2, bottomRotationPoint, bottomRotationBeforePoint1, bottomRotationBeforePoint2]
        let afterPoints = [topRotationPoint, topRotationAfterPoint1, topRotationAfterPoint2, bottomRotationPoint, bottomRotationAfterPoint1, bottomRotationAfterPoint2]
        let homography = Calib3d.findHomography(srcPoints: Converters.vector_Point2f_to_Mat(beforePoints), dstPoints: Converters.vector_Point2f_to_Mat(afterPoints))
        return (homography, midAngle, (topRotationPoint + bottomRotationPoint) / 2)
    }

    func expand(_ m: Mat) -> Mat {
        if m.size() == SIZE_3x3 {
            return m
        }
        let ret = Mat.eye(size: SIZE_3x3, type: m.type())
        m.copy(to: ret.submat(roi: RECT_3x2))
        return ret
    }

    func contract(_ m: Mat) -> Mat {
        return m.submat(roi: RECT_3x2)
    }

    func translate(mat: Mat, xShift: Double, yShift: Double) {
        mat.at(row: 0,col: 2).v += xShift
        mat.at(row: 1,col: 2).v += yShift
    }

    func buildTransforms(transform: Mat, rotationAngle: Double, rotationCenter: Point2f, sizeIn: Size) -> (Mat, Size, Mat) {
        let rotation = expand(Imgproc.getRotationMatrix2D(center: rotationCenter, angle: rotationAngle, scale: 1.0))
        let expandedTransform = expand(transform)
        let fullTransform = expandedTransform * rotation
        let vertices = [Point(x: 0, y: 0), Point(x: 0, y: sizeIn.height), Point(x: sizeIn.width, y: sizeIn.height), Point(x: sizeIn.width, y: 0)].map { Point2f($0) }
        let transformedVertices = transformPoints(points: vertices, transform: fullTransform)
        let xVals = transformedVertices.map { $0.x }
        let yVals = transformedVertices.map { $0.y }
        let bbox = Rect2f(point: Point2f(x: xVals.min()!, y: yVals.min()!), point: Point2f(x: xVals.max()!, y: yVals.max()!))
        translate(mat: fullTransform, xShift: Double(-bbox.x), yShift: Double(-bbox.y))

        let inverseTransform = fullTransform.inv()
        let rightSizedTransform = transform.size() == SIZE_3x3 ? fullTransform : contract(fullTransform)
        let rightSizedInverseTransform = transform.size() == SIZE_3x3 ? inverseTransform : contract(inverseTransform)
        return (rightSizedTransform, Size(bbox.size()), rightSizedInverseTransform)
    }
    
    func transformMat(src: Mat, transform: Mat, bboxSize: Size, borderMode: BorderTypes = .BORDER_CONSTANT) -> Mat {
        let dst = Mat()
        if transform.size() == SIZE_3x3 {
            Imgproc.warpPerspective(src: src, dst: dst, M: transform, dsize: bboxSize, flags: InterpolationFlags.INTER_AREA.rawValue, borderMode: borderMode, borderValue: Scalar(255, 255, 255, 255))
        } else {
            Imgproc.warpAffine(src: src, dst: dst, M: transform, dsize: bboxSize, flags: InterpolationFlags.INTER_AREA.rawValue, borderMode: borderMode, borderValue: Scalar(255, 255, 255, 255))
        }
        return dst
    }

    func getArrowKernel() -> Mat {
        let arrowKernel = Mat(rows: 5, cols: 3, type: CvType.CV_32F)
        let twoOverRoot5 = 2/sqrt(Float(5))
        let kernelData:[Float] = [0, -twoOverRoot5 - 2, -1, -twoOverRoot5, 0, 0, -1, 4, 3 + 4 * twoOverRoot5, -twoOverRoot5, 0, 0, 0, -twoOverRoot5 - 2, -1]
        try! arrowKernel.put(row: 0, col: 0, data: kernelData)
        return arrowKernel
    }

    func getStrokeWidthEstimate(src: Mat, extremes: [Point]) -> Int {
        var count = 0, total = 0
        for extreme in extremes[0..<40] {
            let rightGap = Int(src.width() - extreme.x)
            let left = Int(extreme.x) > 0 ? Int(extreme.x - 1) : 0, right = rightGap >= 25 ? Int(extreme.x) + 25 : Int(src.width() - 1)
            var rightPixels = [UInt8](repeating: 0, count: right - left)
            try! src.get(row: extreme.y, col: extreme.x, data: &rightPixels)
            let initVal = Double(rightPixels[0])
            var foundLeft = false
            for (index, pixel) in rightPixels.dropFirst().enumerated() {
                if !foundLeft && Double(pixel) < initVal * 0.5 {
                    foundLeft = true
                    continue
                } else if foundLeft && Double(pixel) > initVal * 0.7 {
                    count += 1
                    total += index
                    break
                }
            }
        }
        return count != 0 ? (total / count) : -1
    }

    func getExtremePairs(src: Mat, extremes: [Point], rightPairScanBoxOffset: Int, rightPairScanBoxWidth: Int) -> [(Point, Point)] {
        var extremePairs: [(Point, Point)] = []
        outerExtreme: for extreme in extremes {
            let extremeVal: Float = src.at(row: extreme.y, col: extreme.x).v
            for extreme2 in extremes {
                if extreme2 == extreme {
                    break
                }
                if abs(extreme.x - extreme2.x) <= 1 && abs(extreme.y - extreme2.y) <= 1 {
                    continue outerExtreme
                }
            }
            if extreme.y == 0 || extreme.y == src.height() - 1 || extreme.x > src.width() - Int32(rightPairScanBoxOffset) {
                continue
            }
            let minMaxLoc = Core.minMaxLoc(src.safeSubmat(roi: Rect(x: extreme.x + Int32(rightPairScanBoxOffset), y: extreme.y-1, width: Int32(rightPairScanBoxWidth), height: 3)))
            if minMaxLoc.maxVal > Double(-extremeVal) * 0.5 {
                extremePairs.append((extreme, Point(x: extreme.x + Int32(rightPairScanBoxOffset) + minMaxLoc.maxLoc.x, y: extreme.y - 1 + minMaxLoc.maxLoc.y)))
            }
        }
        return extremePairs
    }

    func getRanges(src: Mat, extremePairs: [(Point, Point)]) -> [(Int32, Int32)] {
        var ranges = [(Int32, Int32)]()
        for extremePair in extremePairs {
            let extremeX = extremePair.0.x
            let extremeY = (extremePair.0.y + extremePair.1.y) / 2
            let width:Int32 = 40
            let minMaxLoc = Core.minMaxLoc(src.safeSubmat(roi: Rect(x: extremeX, y: extremeY, width: width, height: 1)))
            let whiteThreshold = (minMaxLoc.maxVal + minMaxLoc.minVal) / 2
            var yLo: Int32 = 0, yHi: Int32 = src.height() - 1
            for y in (0..<extremeY).reversed() {
                let minMaxLoc = Core.minMaxLoc(src.safeSubmat(roi: Rect(x: extremeX, y: y, width: width, height: 1)))
                if minMaxLoc.minVal > whiteThreshold {
                    yLo = y + 1
                    break
                }
            }
            for y in extremeY+1..<src.height() {
                let minMaxLoc = Core.minMaxLoc(src.safeSubmat(roi: Rect(x: extremeX, y: y, width: width, height: 1)))
                if minMaxLoc.minVal > whiteThreshold {
                    yHi = y - 1
                    break
                }
            }
            ranges.append((yLo, yHi))
        }
        ranges.sort { $0.1 - $0.0 > $1.1 - $1.0 }
        return ranges
    }

    func mergeRanges(ranges: [(Int32, Int32)] ) -> [(Int32, Int32, Int)] {
        let average = Double(ranges.map { $0.1 - $0.0 }.reduce(0, +)) / Double(ranges.count)
        let sortedFilteredRanges = ranges.drop { Double($0.1 - $0.0) > average * 2.2 }

        var mergedRanges = [(Int32, Int32, Int)]()
        outerRange : for range in sortedFilteredRanges {
            for (index, mergedRange) in mergedRanges.enumerated() {
                if (mergedRange.0 <= range.0 && range.0 <= mergedRange.1) || (mergedRange.0 <= range.1 && range.1 <= mergedRange.1)  {
                    mergedRanges[index].0 = min(mergedRange.0, range.0)
                    mergedRanges[index].1 = max(mergedRange.1, range.1)
                    mergedRanges[index].2 += 1
                    continue outerRange
                }
            }
            mergedRanges.append((range.0, range.1, 1))
        }

        mergedRanges.sort { $0.0 < $1.0 }
        return mergedRanges
    }

    func getRollupScore(range: IntRange, scores:[Int]) -> Int {
        return scores[range].reduce(0, +)
    }

    func findTextHEdge(start: Int32, end: Int32, minEnd: Int32, mins:[UInt8], maxs:[UInt8], startThreshold: Double) -> Int32 {
        var threshold = startThreshold
        var currentRunLength:Int32 = 0
        var runCount = 0, averageRun = 0.0
        var minMaxs = [(UInt8, UInt8)]()
        var inDark = true
        var lastInDark = start
        var secondLastInDark = start

        for x:Int32 in stride(from: start, to: end, by: start > end ? -1 : 1) {
            if Double(mins[Int(x)]) > threshold {
                if inDark {
                    secondLastInDark = x - 1
                    inDark = false
                }
                currentRunLength += 1
                if runCount > 0 && ((x > minEnd) == (end > start)) && Double(currentRunLength) > 2.5 * averageRun {
                    return lastInDark
                }
            } else {
                if !inDark {
                    if currentRunLength > 0 {
                        if runCount > 0 {
                            threshold = minMaxs.map { Double($0.0) + Double($0.1) }.reduce(0.0, +) / Double(2 * minMaxs.count)
                            minMaxs.removeAll()
                        }
                        averageRun = (Double(currentRunLength) + Double(runCount) * averageRun) / Double(runCount + 1)
                        runCount += 1
                        currentRunLength = 0
                    }
                    inDark = true
                }
                lastInDark = x
                currentRunLength += 1
                if runCount > 0 && ((x > minEnd) == (end > start)) && Double(currentRunLength) > 2.5 * averageRun {
                    return secondLastInDark
                }
                minMaxs.append((mins[Int(x)], maxs[Int(x)]))
            }
        }
        if inDark {
            return secondLastInDark
        } else {
            return lastInDark
        }
    }

    func findTextVEdge(src: Mat, left: Int32, right: Int32, yEst: Int32, isUp: Bool) -> Int32 {
        let startMinMaxLoc = Core.minMaxLoc(src.safeSubmat(roi: Rect(x: left, y: yEst + Int32(isUp ? 1 : -1) , width: right - left, height: 1)))
        let threshold = (startMinMaxLoc.minVal + startMinMaxLoc.maxVal * 4) / 5

        for y:Int32 in stride(from: yEst + Int32(isUp ? -1 : 1), to: isUp ? 0 : src.height() - 1, by: isUp ? -1 : 1) {
            let dst = Mat()
            Core.reduce(src: src.safeSubmat(roi: Rect(x: left, y: y, width: right - left, height: 1)), dst: dst, dim: 1, rtype: Core.REDUCE_AVG)
            let avg: UInt8 = dst.at(row: 0, col: 0).v
            if Double(avg) > threshold {
                return y
            }
        }
        return isUp ? 0 : src.height() - 1
    }

    func calcRowDims(src: Mat, excludeArea: [Point2d], points: [Point], yStartEst: Int32, yEndEst: Int32) -> Rect {
        let dstMinMat = Mat(), dstMaxMat = Mat()
        var dstMin = [UInt8](repeating: 0, count: Int(src.width())), dstMax = [UInt8](repeating: 0, count: Int(src.width()))
        Core.reduce(src: src.submat(roi: Rect(x: 0, y: yStartEst, width: src.width(), height: yEndEst - yStartEst)), dst: dstMinMat, dim: 0, rtype: Core.REDUCE_MIN)
        try! dstMinMat.get(row: 0, col: 0, data: &dstMin)
        Core.reduce(src: src.submat(roi: Rect(x: 0, y: yStartEst, width: src.width(), height: yEndEst - yStartEst)), dst: dstMaxMat, dim: 0, rtype: Core.REDUCE_MAX)
        try! dstMaxMat.get(row: 0, col: 0, data: &dstMax)
        let xSortedPoints = points.sorted { $0.x < $1.x }
        let startLeftwardIndex = xSortedPoints.count > 1 ? 1 : 0
        let startLeftward = xSortedPoints[startLeftwardIndex]
        let minEndLeftward = xSortedPoints[max(startLeftwardIndex - 1, 0)]
        let startRightwardIndex = xSortedPoints.count > 1 ? xSortedPoints.count - 2 : 0
        let startRightward = xSortedPoints[startRightwardIndex]
        let minEndRightward = xSortedPoints[min(startRightwardIndex + 1, xSortedPoints.count - 1)]
        let startLeftMinMaxLoc = Core.minMaxLoc(src.safeSubmat(roi: Rect(x: startLeftward.x - 5, y: startLeftward.y - 5, width: 10, height: 10)))
        let startLeftThreshold = (startLeftMinMaxLoc.minVal + startLeftMinMaxLoc.maxVal) / 2
        let rangeStartEst = ExtremeCalculator.getThresholds(excludeArea, y: Double(yStartEst))
        let rangeEndEst = ExtremeCalculator.getThresholds(excludeArea, y: Double(yEndEst))
        if rangeStartEst == nil || rangeEndEst == nil {
            return NULL_RECT
        }
        let left = findTextHEdge(start: startLeftward.x - 1, end: max(rangeStartEst!.start, rangeEndEst!.start), minEnd: minEndLeftward.x, mins: dstMin, maxs: dstMax, startThreshold: startLeftThreshold)
        let startRightMinMaxLoc = Core.minMaxLoc(src.safeSubmat(roi: Rect(x: startRightward.x - 5, y: startRightward.y - 5, width: 10, height: 10)))
        let startRightThreshold = (startRightMinMaxLoc.minVal + startRightMinMaxLoc.maxVal) / 2
        let right = findTextHEdge(start: startRightward.x + 1, end: min(rangeStartEst!.end, rangeEndEst!.end), minEnd: minEndRightward.x, mins: dstMin, maxs: dstMax, startThreshold: startRightThreshold)
        if left >= right {
            return NULL_RECT
        }
        let top = findTextVEdge(src: src, left: left, right: right, yEst: yStartEst, isUp: true)
        let bottom = findTextVEdge(src: src, left: left, right: right, yEst: yEndEst, isUp: false)
        return Rect(point: Point(x: left, y: top), point: Point(x: right, y: bottom))
    }

    func gapsOk(gaps: [Double], heights: [Double]) -> Bool {
        let gapsSorted = gaps.sorted().map { $0 + 10 }
        if abs(gapsSorted[gaps.count - 1] - gapsSorted[0])/gapsSorted[gaps.count - 1] > 0.22 {
            return false
        }
        let heightsSorted = heights.sorted().map { $0 + 10 }
        if abs(heightsSorted[heights.count - 1] - heightsSorted[0])/heightsSorted[gaps.count - 1] > 0.22 {
            return false
        }
        for triple in zip(gaps, zip(heights, heights.dropFirst())) {
            if triple.0 > (triple.1.0 + triple.1.1) {
                return false
            }
        }
        return true
    }

    func rangeOverlap(range1: (Int32, Int32), range2: (Int32, Int32)) -> (Int32, Int32) {
        if range1.1 <= range2.0 || range2.1 <= range1.0  {
            return (0, 0)
        } else {
            return (max(range1.0, range2.0), min(range1.1, range2.1))
        }
    }

    func horizontalAlignmentOk(lefts: [Int32], widths: [Int32]) -> Bool {
        let ranges = zip(lefts, widths).map { ($0.0, $0.0 + $0.1) }
        let overlap = ranges.reduce((Int32.min, Int32.max), rangeOverlap)
        let widest = Double(widths.max()!)
        return Double(overlap.1 - overlap.0)/widest > 0.78
    }

    func findBestCandidate(rollupScoreMap: [(IntRange, Int)], getRectFunc: (Int)->Rect) -> ((IntRange, Int)?, [Rect]) {
        var rects = [Int:Rect]()
        for candidate in rollupScoreMap {
            candidate.0.forEach { if rects[$0] == nil { rects[$0] = getRectFunc($0) } }
            if candidate.0.contains(where: { rects[$0] == NULL_RECT }) {
                continue
            }
            let widths = candidate.0.map { rects[$0]!.width }
            let lefts = candidate.0.map { rects[$0]!.x }
            let heights = candidate.0.map { Double(rects[$0]!.height) }
            let yStarts = candidate.0.map { rects[$0]!.y }.dropFirst()
            let yEnds = candidate.0.map { rects[$0]!.y + rects[$0]!.height }.dropLast()
            let gaps = zip(yStarts, yEnds).map { Double($0.0 - $0.1) }
            if candidate.0.count == 3 {
                if gapsOk(gaps: gaps, heights: heights) && horizontalAlignmentOk(lefts: lefts, widths: widths) {
                    return (candidate, candidate.0.map { rects[$0]! })
                }
            } else {
                if gapsOk(gaps: gaps, heights: heights) && horizontalAlignmentOk(lefts: lefts, widths: widths) {
                    return (candidate, candidate.0.map { rects[$0]! })
                }
            }
        }
        return (nil,[])
    }

    public func getMrz(src: Mat) -> Quadrilateral? {
        let srcGray = Mat()
        let width = Double(src.width())
        Imgproc.cvtColor(src: src, dst: srcGray, code: .COLOR_BGR2GRAY)
        
        let srcGrayCorrected = brightnessCorrection(srcGray)
        let goodFeatures = getGoodFeatures(srcGrayCorrected)

        let adjustedAngle = getTextOrientation(points: goodFeatures, width: width)

        let rotation = Imgproc.getRotationMatrix2D(center: Point2f(x: Float(src.width()) / 2, y: Float(src.height()) / 2), angle: Double(-adjustedAngle), scale: 1.0)

        var m1x = [Double](repeating: 0, count: 3)
        try! rotation.get(row: 1, col: 0, data: &m1x)

        let point2RotatedY = goodFeatures.map {
            ($0, Double(m1x[0] * Double($0.x) + m1x[1] * Double($0.y)))
        }

        let (topWarpAngle, topCenter) = getWarpData(points: point2RotatedY.filter { $0.1 < Double(src.height() / 3) - m1x[2] }.map { $0.0 }, width: width)

        let (bottomWarpAngle, bottomCenter) = getWarpData(points:point2RotatedY.filter { $0.1 > Double(2 * src.height() / 3) - m1x[2] }.map { $0.0 }, width: width)

        let angleDiff = adjustAngle(max(topWarpAngle, bottomWarpAngle) - min(topWarpAngle, bottomWarpAngle))
        let srcSize = src.size()
        
        let (dewarpMat, rotationAngle, rotationCenter) = (angleDiff >= 1.2 && angleDiff <= 19) ?
            getDewarpTransform(topWarpAngle: topWarpAngle, topCenter: topCenter, bottomWarpAngle: bottomWarpAngle, bottomCenter: bottomCenter) :
            (ID_3x2D.clone(), Double(adjustedAngle), Point2f(x: Float(srcSize.width) / 2.0, y: Float(srcSize.height) / 2.0))
        let (transMat, bboxSize, untransMat) = buildTransforms(transform: dewarpMat, rotationAngle: -rotationAngle, rotationCenter: rotationCenter, sizeIn: srcSize)

        let widthDouble = Double(src.size().width), heightDouble = Double(src.size().height)

        let transformedVertexPoints = transformPoints(points: [Point2d(x: 5.0, y: 5.0), Point2d(x: 5.0, y: heightDouble - 5.0), Point2d(x: widthDouble - 5.0, y: heightDouble - 5.0), Point2d(x: widthDouble - 5.0, y: 5.0)], transform: transMat)
        let transformedGoodFeaturesPoints = transformPoints(points: goodFeatures.map { Point2f($0) }, transform: transMat).map { Point($0) }
        let straightMat = transformMat(src: srcGrayCorrected, transform: transMat, bboxSize: bboxSize)

        let arrowKernel = getArrowKernel()
        let arrowKernelResult = Mat()
        Imgproc.filter2D(src: straightMat, dst: arrowKernelResult, ddepth: CvType.CV_32F, kernel: arrowKernel, anchor: Point(x: 1, y: 2))
        let extremes = ExtremeCalculator.getExtremes(arrowKernelResult, excludeArea: transformedVertexPoints, threshold: extremeThreshold, minPointCount: extremeCount)

        let strokeWidthEstimate = getStrokeWidthEstimate(src: straightMat, extremes: extremes)
        guard strokeWidthEstimate != -1 else {
            return nil
        }
        let rightPairScanBoxOffset = strokeWidthEstimate < 4 ? strokeWidthEstimate : (strokeWidthEstimate * 3) / 4
        let rightPairScanBoxWidth = strokeWidthEstimate < 8 ? 3 : strokeWidthEstimate / 2

        let extremePairs = getExtremePairs(src: arrowKernelResult, extremes: extremes, rightPairScanBoxOffset: rightPairScanBoxOffset, rightPairScanBoxWidth: rightPairScanBoxWidth)
        
        let ranges = getRanges(src: straightMat, extremePairs: extremePairs)
        let mergedRanges = mergeRanges(ranges: ranges)
        guard mergedRanges.count >= 2 else {
            return nil
        }

        let scores = mergedRanges.map{ $0.2 }
        let candidateRanges = mergedRanges.indices.dropFirst(2).map { ($0-2...$0) } + mergedRanges.indices.dropFirst(1).map { ($0-1...$0) }
        let rollupScoreMap = candidateRanges.map { ($0, getRollupScore(range: $0, scores: scores)) }.sorted { $1.1 < $0.1 }

        func getRect(index: Int) -> Rect {
            calcRowDims(src: straightMat, excludeArea: transformedVertexPoints, points: extremePairs.map { Point(x: ($0.0.x + $0.1.x) / 2, y:  ($0.0.y + $0.1.y)  / 2) }.filter { $0.y >= mergedRanges[index].0 && $0.y <= mergedRanges[index].1 }, yStartEst: mergedRanges[index].0, yEndEst: mergedRanges[index].1)
        }

        let (bestCandidate, rects) = findBestCandidate(rollupScoreMap: rollupScoreMap, getRectFunc: getRect)
        
        guard bestCandidate != nil else {
            print("MRZ not found")
            return nil
        }

        let pointsInBestCandidate = transformedGoodFeaturesPoints.filter { point in
            rects.map { Rect(x: $0.x - 5, y: $0.y - 5, width: $0.width + 10, height: $0.height + 10) }.contains { $0.contains(point) }
        }

        let adjustedTextAngle = getTextOrientation(points: pointsInBestCandidate, width: width)

        let xMin = rects.map { $0.x }.min()!, xMax = rects.map { $0.x + $0.width }.max()!
        let yMin = rects.map { $0.y }.min()!, yMax = rects.map { $0.y + $0.height }.max()!
        let xAdjust = (xMax - xMin) / 30, yAdjust = ((yMax - yMin) / 10) + Int32(abs(sin(adjustedTextAngle * Float.pi / 180.0)) * Float(xMax - xMin)/2)
        let mrzCornersUntransformed = [Point(x: xMin - xAdjust, y: yMin - yAdjust), Point(x: xMax + xAdjust, y: yMin - yAdjust), Point(x: xMax + xAdjust, y: yMax + yAdjust), Point(x: xMin - xAdjust, y: yMax + yAdjust)]
        let mrzCorners = transformPoints(points: mrzCornersUntransformed.map { Point2d($0) }, transform: untransMat).map { Point($0) }
        let adjustMat = Imgproc.getRotationMatrix2D(center: Point2f(mrzCorners[0] + mrzCorners[2]) / 2, angle: Double(adjustedTextAngle), scale: 1.0)
        let mrzCornersAdjusted = transformPoints(points: mrzCorners.map { Point2d($0) }, transform: adjustMat)
        return Quadrilateral(p1: mrzCornersAdjusted[0], p2: mrzCornersAdjusted[1], p3: mrzCornersAdjusted[2], p4: mrzCornersAdjusted[3])
    }
}
