#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <set>

#include <android/log.h>

#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "ExtremeCalculator", __VA_ARGS__))
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "ExtremeCalculator", __VA_ARGS__))

/// throw java exception
static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method) {
    std::string what = "unknown exception";
    jclass je = 0;

    if(e) {
        std::string exception_type = "std::exception";

        if(dynamic_cast<const cv::Exception*>(e)) {
            exception_type = "cv::Exception";
            je = env->FindClass("org/opencv/core/CvException");
        }

        what = exception_type + ": " + e->what();
    }

    if(!je) je = env->FindClass("java/lang/Exception");
    env->ThrowNew(je, what.c_str());

    LOGE("%s caught %s", method, what.c_str());
    CV_UNUSED(method);        // avoid "unused" warning
}

float eval(cv::Mat* mat, int i) {
    const int width = mat->cols;
    return mat->at<float>(i / width, i % width);
}

std::vector<double> getThresholds(double y, std::vector<cv::Point2d>& corners) {
    std::vector<double> thresholds;

    for (int cornerIndex = 0; cornerIndex < corners.size(); cornerIndex++) {
        cv::Point2d corner1 = corners[cornerIndex], corner2 = corners[(cornerIndex + 1) % corners.size()];
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

extern "C"
JNIEXPORT jlong JNICALL
Java_net_telepathix_mrzdetector_ExtremeCalculatorKt_nGetExtremes(JNIEnv *env, jclass clazz,
                                                                 jlong src,
                                                                 jlong exclude_area_corners,
                                                                 jdouble threshold,
                                                                 jint min_point_count) {
    static const char method_name[] = "ExtremeCalculator::nGetExtremes";
    try {
        LOGD("%s", method_name);
        auto mat = (cv::Mat*) src;
        auto corners = (cv::Mat*) exclude_area_corners;
        std::vector<cv::Point2d> excludePolygon = (std::vector<cv::Point2d>)(*corners);

        const int width = mat->cols;
        const int height = mat->rows;

        auto cmp = [&mat](int a, int b) { float aVal = eval(mat, a), bVal = eval(mat, b); return aVal != bVal ? aVal > bVal : a < b; };
        std::set<int, decltype(cmp)> extremes(cmp);

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
                if (extremes.size() < min_point_count || eval(mat, offset) < threshold) {
                    extremes.insert(offset);
                } else {
                    if (eval(mat, offset) < eval(mat, *extremes.begin())) {
                        extremes.erase(*extremes.begin());
                        extremes.insert(offset);
                    }
                }
            }
        }
        while (extremes.size() > min_point_count && eval(mat, *extremes.begin()) > threshold) {
            extremes.erase(*extremes.begin());
        }
        std::vector<cv::Point> extremePoints;
        for (auto extreme : extremes) {
            extremePoints.push_back(cv::Point(extreme % width, extreme / width));
        }
        jlong ret = (jlong) new cv::Mat(extremePoints, true);
        return ret;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}