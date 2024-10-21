package com.barkrn

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableMap

abstract class BarkRnSpec internal constructor(context: ReactApplicationContext) :
  ReactContextBaseJavaModule(context) {

  abstract fun init_context(model_path: String, params: ReadableMap, promise: Promise)
  abstract fun generate(id: Int, text: String, out_path: String, threads: Int, promise: Promise)
  abstract fun release_context(id: Int, promise: Promise)
  abstract fun release_all_contexts(promise: Promise)
}
