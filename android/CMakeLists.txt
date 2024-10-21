cmake_minimum_required(VERSION 3.4.1)
project(BarkRn)

set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(BARK_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)

FetchContent_Declare(
    BARK_CPP
    GIT_REPOSITORY https://github.com/PABannier/bark.cpp.git
    GIT_TAG        v1.0.0
    GIT_PROGRESS   TRUE
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(BARK_CPP)

add_library(bark-rn            SHARED
            ../cpp/utils.cpp
            cpp-adapter.cpp
)

# Specifies a path to native header files.
include_directories(
            ../cpp
            ${BARK_CPP_SOURCE_DIR}
)

target_link_libraries(bark-rn PUBLIC bark)
