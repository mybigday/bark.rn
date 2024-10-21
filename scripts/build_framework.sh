#!/bin/bash

set -x
set -e

VERSION="v1.0.0"
SOURCE="https://github.com/PABannier/bark.cpp.git"
ROOT=$(realpath $(dirname $0))

PLATFORMS=("OS64COMBINED" "TVOSCOMBINED")

# build xcframework from https://github.com/PABannier/bark.cpp for iOS

tmpdir=$(mktemp -d)

# remove tmpdir on exit or other termination signals
trap "rm -rf $tmpdir" EXIT INT TERM

git clone $SOURCE $tmpdir
cd $tmpdir
git checkout $VERSION
git submodule update --init --recursive

# patch CMakeLists.txt, set BARK_BUILD_EXAMPLES to OFF
sed -i '' 's/set(BARK_BUILD_EXAMPLES ON)/set(BARK_BUILD_EXAMPLES OFF)/' CMakeLists.txt

cat <<'EOF' >> CMakeLists.txt
if (BUILD_FRAMEWORK)
  file(GLOB HEADERS "spm-headers/*.h")
  set_target_properties(${BARK_LIB} PROPERTIES FRAMEWORK TRUE
    PUBLIC_HEADER "${HEADERS}")
endif()
EOF

FRAMEWORKS=()
for platform in "${PLATFORMS[@]}"; do
  plat_slug=$(echo $platform | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
  case $platform in
    "OS64COMBINED")
      release_dir="Release-iphoneos"
      ;;
    "TVOSCOMBINED")
      release_dir="Release-appletvos"
      ;;
  esac

  cmake . -G Xcode -B build-$plat_slug \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DCMAKE_TOOLCHAIN_FILE=$ROOT/scripts/ios.toolchain.cmake \
    -DPLATFORM=$platform \
    -DBUILD_FRAMEWORK=ON \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
  cmake --build build-$plat_slug --config Release -j $(sysctl -n hw.logicalcpu) -- CODE_SIGNING_ALLOWED=NO
  FRAMEWORKS+=("build-$plat_slug/$release_dir/bark.framework")
done

ARGS=()
for framework in "${FRAMEWORKS[@]}"; do
  ARGS+=("-framework $framework")
done

xcodebuild -create-xcframework ${ARGS[@]} -output $ROOT/bark.xcframework

