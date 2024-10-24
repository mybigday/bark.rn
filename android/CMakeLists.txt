cmake_minimum_required(VERSION 3.4.1)
project(BarkRn)

set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_STANDARD 17)



if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma -mf16c -mavx -mavx2")
endif()

add_library(bark-rn            SHARED
            ../cpp/utils.h
            ../cpp/utils.cpp
            ../cpp/dr_wav.h
            ../cpp/ggml.h
            ../cpp/ggml.c
            ../cpp/ggml-alloc.h
            ../cpp/ggml-alloc.c
            ../cpp/ggml-backend.h
            ../cpp/ggml-backend.c
            ../cpp/bark.h
            ../cpp/bark.cpp
            ../cpp/encodec.h
            ../cpp/encodec.cpp
            cpp-adapter.cpp
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(bark-rn PRIVATE -march=armv8.2-a+fp16)
endif()

target_compile_options(bark-rn PRIVATE -O3 -DNDEBUG -pthread)
# target_compile_options(bark-rn PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
# target_compile_options(bark-rn PRIVATE -ffunction-sections -fdata-sections)

target_link_options(bark-rn PRIVATE -Wl,--gc-sections)
target_link_options(bark-rn PRIVATE -Wl,--exclude-libs,ALL)
target_link_options(bark-rn PRIVATE -flto)

# Specifies a path to native header files.
include_directories(
            ../cpp
)

target_link_libraries(bark-rn PUBLIC log)
