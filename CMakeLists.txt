project("main")
cmake_minimum_required( VERSION 2.8.8 )

set(SDK_DIR /home/tronlong/rv1126b-linux-SDK-20250814)
set(CMAKE_CXX_COMPILER ${SDK_DIR}/buildroot/output/rockchip_rv1126b/host/bin/aarch64-linux-g++)
set(CMAKE_C_COMPILER ${SDK_DIR}/buildroot/output/rockchip_rv1126b/host/bin/aarch64-linux-gcc)
set(OpenCV_DIR ${SDK_DIR}/buildroot/output/rockchip_rv1126b/build/opencv4-4.9.0/buildroot-build/)
find_package(OpenCV REQUIRED)

add_executable(main
    main.cpp
    algo/cbisp.cpp
)   

# 添加头文件
include_directories(${CMAKE_SYSROOT})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/algo)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加库文件
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_definitions(-O3 -pthread -lm -lrt -std=c++17 -mcpu=cortex-a53  -mtune=cortex-a72 -Wall
                -ftree-vectorize -ffast-math -march=armv8-a+simd -funroll-loops -D_GNU_SOURCE )

target_link_libraries(main 
                        pthread 
                        ${OpenCV_LIBS} 
                        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcompress.a
                    )