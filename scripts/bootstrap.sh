#!/bin/bash

set -x
set -e

ROOT=$(realpath $(dirname $(dirname $0)))

cp \
  $ROOT/bark.cpp/bark.h \
  $ROOT/bark.cpp/bark.cpp \
  $ROOT/bark.cpp/encodec.cpp/encodec.cpp \
  $ROOT/bark.cpp/encodec.cpp/encodec.h \
  $ROOT/bark.cpp/encodec.cpp/ggml/include/ggml/ggml.h \
  $ROOT/bark.cpp/encodec.cpp/ggml/include/ggml/ggml-alloc.h \
  $ROOT/bark.cpp/encodec.cpp/ggml/include/ggml/ggml-backend.h \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml.c \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml-alloc.c \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml-backend.c \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml-metal.h \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml-metal.m \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml-metal.metal \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml-opencl.cpp \
  $ROOT/bark.cpp/encodec.cpp/ggml/src/ggml-opencl.h \
  $ROOT/cpp
