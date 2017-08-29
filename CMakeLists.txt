cmake_minimum_required(VERSION 2.8)

project(FGS)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(OpenCV_DIR /home/andy/opencv/build/)
set(OpenCV_FOUND 1)
find_package(OpenCV REQUIRED)

set(SRC_FOLDER "src")
file(GLOB SOURCES "${SRC_FOLDER}/*.cpp")

add_executable(fgs ${SOURCES})
target_link_libraries(fgs ${OpenCV_LIBS})