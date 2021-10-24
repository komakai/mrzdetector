//
//  ExtremeCalculator.m
//  MRZDetect
//
//  Created by Giles Payne on 2021/05/24.
//

#import "ExtremeCalculator.h"
#import <opencv2/Point2i.h>
#import <opencv2/Point2d.h>
#import <opencv2/Range.h>
#import <opencv2/Mat.h>
#import <set>
#import <iostream>

@implementation ExtremeCalculator

float eval(cv::Mat& mat, int i) {
    const int width = mat.cols;
    return mat.at<float>(i / width, i % width);
}

std::vector<double> getThresholds(double y, std::vector<Point2d*>& corners) {
    std::vector<double> thresholds;

    for (int cornerIndex = 0; cornerIndex < corners.size(); cornerIndex++) {
        Point2d *corner1 = corners[cornerIndex], *corner2 = corners[(cornerIndex + 1) % corners.size()];
        std::vector<double> yValues{ y, corner1.y, corner2.y };
        std::sort(yValues.begin(), yValues.end());
        if (yValues[1] == y) {
            if (corner1.y == corner2.y) {
                continue;
            }
            double grad = (corner2.x - corner1.x) / (corner2.y - corner1.y);
            double intersectionX = corner1.x + (y - corner1.y) * grad;
            thresholds.push_back(intersectionX);
        }
    }

    std::sort(thresholds.begin(), thresholds.end());
    return thresholds;
}

bool withinThresholds(double x, std::vector<double> thresholds) {
    return std::count_if(thresholds.begin(), thresholds.end(), [&x](double d){return d<x;}) % 2 == 1;
}

+(nullable Range*)getThresholds:(NSArray<Point2d*>*)excludeAreaCorners y:(double)y {
    std::vector<Point2d*> excludePolygon;
    for (Point2d* excludeAreaCorner in excludeAreaCorners) {
        excludePolygon.push_back(excludeAreaCorner);
    }
    auto thresholds = getThresholds((double)y, excludePolygon);
    return thresholds.size() >= 2 ? [[Range alloc] initWithStart:floor(thresholds.front()) end:ceil(thresholds.back())] : nil;
}

+(NSArray<Point2i*>*)getExtremes:(Mat*)mat excludeArea:(NSArray<Point2d*>*)excludeAreaCorners threshold:(float)threshold minPointCount:(int)minPointCount {
    const int width = mat.width;
    const int height = mat.height;

    auto cmp = [&mat](int a, int b) { float aVal = eval(mat.nativeRef, a), bVal = eval(mat.nativeRef, b); return aVal != bVal ? aVal > bVal : a < b; };
    std::set<int, decltype(cmp)> extremes(cmp);
    
    std::vector<Point2d*> excludePolygon;
    for (Point2d* excludeAreaCorner in excludeAreaCorners) {
        excludePolygon.push_back(excludeAreaCorner);
    }

    for (int row = 0; row < height; row++) {
        auto thresholds = getThresholds((double)row, excludePolygon);
        if (thresholds.size() == 0) {
            continue;
        }
        for (int col = 0; col < width; col++) {
            int offset = row * width + col;
            if (!withinThresholds((double)col, thresholds)) {
                continue;
            }
            if (extremes.size() < minPointCount || eval(mat.nativeRef, offset) < threshold) {
                extremes.insert(offset);
            } else {
                if (eval(mat.nativeRef, offset) < eval(mat.nativeRef, *extremes.begin())) {
                    extremes.erase(*extremes.begin());
                    extremes.insert(offset);
                }
            }
        }
    }
    while (extremes.size() > minPointCount && eval(mat.nativeRef, *extremes.begin()) > threshold) {
        extremes.erase(*extremes.begin());
    }
    NSMutableArray *ret = [NSMutableArray arrayWithCapacity:extremes.size()];
    for (auto extreme : extremes) {
        [ret addObject:[[Point2i alloc] initWithX:extreme % width y:extreme / width]];
    }
    return ret;
}

@end
