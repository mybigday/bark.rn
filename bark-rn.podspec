require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
folly_compiler_flags = '-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1 -Wno-comma -Wno-shorten-64-to-32'

base_compiler_flags = "-DWSP_GGML_USE_ACCELERATE -Wno-shorten-64-to-32"
folly_compiler_flags = "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1 -Wno-comma"

# Use base_optimizer_flags = "" for debug builds
# base_optimizer_flags = ""
base_optimizer_flags = "-O3 -DNDEBUG" +
 " -fvisibility=hidden -fvisibility-inlines-hidden" +
 " -ffunction-sections -fdata-sections"

Pod::Spec.new do |s|
  s.name         = "bark-rn"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "11.0", :tvos => "11.0" }
  s.source       = { :git => "https://github.com/mybigday/bark.rn.git", :tag => "#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm}",
                   "cpp/ggml.{c,h}",
                   "cpp/ggml-alloc.{c,h}",
                   "cpp/ggml-backend.{c,h}",
                   "cpp/encodec.{cpp,h}",
                   "cpp/bark.{cpp,h}",
                   "cpp/dr_wav.h",
                   "cpp/utils.{cpp,h}"
  
  s.compiler_flags = base_compiler_flags
  s.pod_target_xcconfig = {
    "OTHER_CFLAGS" => base_optimizer_flags,
    "OTHER_CPLUSPLUSFLAGS" => base_optimizer_flags
  }

  # Use install_modules_dependencies helper to install the dependencies if React Native version >=0.71.0.
  # See https://github.com/facebook/react-native/blob/febf6b7f33fdb4904669f99d795eba4c0f95d7bf/scripts/cocoapods/new_architecture.rb#L79.
  if respond_to?(:install_modules_dependencies, true)
    install_modules_dependencies(s)
  else
    s.dependency "React-Core"

    # Don't install the dependencies when we run `pod install` in the old architecture.
    if ENV['RCT_NEW_ARCH_ENABLED'] == '1' then
      s.compiler_flags = base_compiler_flags + " " + folly_compiler_flags + " -DRCT_NEW_ARCH_ENABLED=1"
      s.pod_target_xcconfig    = {
          "HEADER_SEARCH_PATHS" => "\"$(PODS_ROOT)/boost\"",
          "OTHER_CPLUSPLUSFLAGS" => "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1",
          "CLANG_CXX_LANGUAGE_STANDARD" => "c++17"
      }
      s.dependency "React-Codegen"
      s.dependency "RCT-Folly"
      s.dependency "RCTRequired"
      s.dependency "RCTTypeSafety"
      s.dependency "ReactCommon/turbomodule/core"
    end
  end
end
