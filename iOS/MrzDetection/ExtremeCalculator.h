//
//  ExtremeCalculator.h
//  MRZDetect
//
//  Created by Giles Payne on 2021/05/24.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#endif

@class Point2i;
@class Point2d;
@class Range;
@class Mat;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExtremeCalculator : NSObject

+(NSArray<Point2i*>*)getExtremes:(Mat*)mat excludeArea:(NSArray<Point2d*>*)excludeAreaCorners threshold:(float)threshold minPointCount:(int)minPointCount;
+(nullable Range*)getThresholds:(NSArray<Point2d*>*)excludeAreaCorners y:(double)y;

@end

NS_ASSUME_NONNULL_END
