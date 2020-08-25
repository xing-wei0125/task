cmake_minimum_required(VERSION 3.5)

project(Main)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

add_executable(Main main.cpp rectify.cpp)
target_include_directories(Main PUBLIC ${PROJECT_SOURCE_DIR} $(OpenCV_INCLUDE_DIRS))
target_link_libraries(Main PUBLIC ${OpenCV_LIBS})
