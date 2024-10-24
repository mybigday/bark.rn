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
  cp "$file" "cpp/"
done

patch -p0 < ./scripts/ggml-alloc.c.patch
patch -p0 < ./scripts/ggml.c.patch

if [ "$(uname)" == "Darwin" ]; then
  SED="sed -i ''"
else
  SED="sed -i"
fi

PATCH_LOG_FILES=(
  cpp/encodec.h
  cpp/encodec.cpp
  cpp/bark.h
  cpp/bark.cpp
)

for file in "${PATCH_LOG_FILES[@]}"; do
  filename=$(basename "$file")
  $SED 's/fprintf(stderr, /LOGE(/g' "cpp/$filename"
  $SED 's/printf(/LOGI(/g' "cpp/$filename"
  $SED '/#pragma once/a #include "log.h"' "cpp/$filename"
done

for file in "${FILES[@]}"; do
  filename=$(basename "$file")
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  $SED 's/GGML_/BARK_GGML_/g' "cpp/$filename"
  $SED 's/ggml_/bark_ggml_/g' "cpp/$filename"
  $SED 's/GGUF_/BARK_GGUF_/g' "cpp/$filename"
  $SED 's/gguf_/bark_gguf_/g' "cpp/$filename"
  $SED 's/GGMLMetalClass/BARKGGMLMetalClass/g' "cpp/$filename"
done
