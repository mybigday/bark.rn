cmake_minimum_required(VERSION 3.4.1)
project(BarkRn)

set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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

# Specifies a path to native header files.
include_directories(
            ../cpp
)

target_link_libraries(bark-rn PUBLIC m)
