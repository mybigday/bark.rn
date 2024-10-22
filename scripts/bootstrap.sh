#!/bin/bash

set -x -e

cd $(realpath $(dirname $(dirname $0)))

FILES=(
  bark.cpp/bark.h
  bark.cpp/bark.cpp
  bark.cpp/encodec.cpp/encodec.cpp
  bark.cpp/encodec.cpp/encodec.h
  bark.cpp/encodec.cpp/ggml/include/ggml/ggml.h
  bark.cpp/encodec.cpp/ggml/include/ggml/ggml-alloc.h
  bark.cpp/encodec.cpp/ggml/include/ggml/ggml-backend.h
  bark.cpp/encodec.cpp/ggml/src/ggml.c
  bark.cpp/encodec.cpp/ggml/src/ggml-alloc.c
  bark.cpp/encodec.cpp/ggml/src/ggml-backend.c
  bark.cpp/encodec.cpp/ggml/src/ggml-metal.h
  bark.cpp/encodec.cpp/ggml/src/ggml-metal.m
  bark.cpp/encodec.cpp/ggml/src/ggml-metal.metal
  bark.cpp/encodec.cpp/ggml/src/ggml-opencl.cpp
  bark.cpp/encodec.cpp/ggml/src/ggml-opencl.h
)

for file in "${FILES[@]}"; do
  cp "$file" "cpp"
done
