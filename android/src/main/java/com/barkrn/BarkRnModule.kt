package com.barkrn

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.Arguments

import java.io.File

class BarkRnModule internal constructor(context: ReactApplicationContext) :
  BarkRnSpec(context) {

  init {
    System.loadLibrary("bark-rn")
  }

  private var next_id: Int = 0
  private var contexts: MutableMap<Int, BarkContext> = mutableMapOf()

  override fun getName(): String {
    return NAME
  }

  override fun invalidate() {
    super.invalidate()
    for (context in contexts.values) {
      context.release()
    }
    contexts.clear()
  }


  @ReactMethod
  override fun init_context(model_path: String, params: ReadableMap, promise: Promise) {
    val id = next_id
    next_id += 1
    contexts[id] = BarkContext(model_path, params.toHashMap())
    promise.resolve(id)
  }

  @ReactMethod
  override fun generate(id: Int, text: String, audio_path: String, threads: Int, promise: Promise) {
    contexts[id]?.let { context ->
      val result = context.generate(text, audio_path, threads)
      val resultMap = Arguments.createMap()
      resultMap.putBoolean("success", result.success)
      resultMap.putInt("load_time", result.load_time)
      resultMap.putInt("eval_time", result.eval_time)
      promise.resolve(resultMap)
    } ?: promise.reject("Context not found")
  }

  @ReactMethod
  override fun release_context(id: Int, promise: Promise) {
    contexts[id]?.let { context ->
      context.release()
      contexts.remove(id)
      promise.resolve(null)
    } ?: promise.reject("Context not found")
  }

  @ReactMethod
  override fun release_all_contexts(promise: Promise) {
    for (context in contexts.values) {
      context.release()
    }
    contexts.clear()
    promise.resolve(null)
  }

  companion object {
    const val NAME = "BarkRn"
  }
}
