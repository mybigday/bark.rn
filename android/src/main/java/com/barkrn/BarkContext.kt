package com.barkrn

class BarkContext {
  private var context: Long = 0L

  class BarkResult(success: Boolean, load_time: Int, eval_time: Int)

  external fun nativeInitContext(model_path: String, params: Map<String, Any>): Long
  external fun nativeGenerate(context: Long, text: String, out_path: String, threads: Int): BarkResult
  external fun nativeReleaseContext(context: Long)

  constructor(model_path: String, params: Map<String, Any>) {
    context = nativeInitContext(model_path, params)
  }

  fun generate(text: String, out_path: String, threads: Int = 1): BarkResult {
    if (context == 0) {
      throw IllegalStateException("Context not initialized")
    }
    return nativeGenerate(context, text, out_path, threads)
  }

  fun release() {
    if (context != 0L) {
      nativeReleaseContext(context)
      context = 0L
    }
  }
}
