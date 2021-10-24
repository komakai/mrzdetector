cmake_minimum_required(VERSION 3.6)

add_library(mrzdetector SHARED ExtremeCalculator.cpp)

include_directories(../../opencv/native/jni/include/)

find_library(log-lib log)

add_library(opencv-lib SHARED IMPORTED)

set_target_properties(opencv-lib PROPERTIES IMPORTED_LOCATION ../../../../../opencv/native/libs/${ANDROID_ABI}/libopencv_java4.so)

target_link_libraries(mrzdetector opencv-lib ${log-lib})
