#include "utils.h"
#include <jni.h>
#include <thread>

template <typename T>
T get_map_value(JNIEnv *env, jobject params, const char *key) {
  jclass map_class = env->FindClass("java/util/Map");
  jmethodID get_method = env->GetMethodID(
      map_class, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
  return env->CallObjectMethod(params, get_method, env->NewStringUTF(key));
}

bool has_map_key(JNIEnv *env, jobject params, const char *key) {
  jclass map_class = env->FindClass("java/util/Map");
  jmethodID contains_key_method =
      env->GetMethodID(map_class, "containsKey", "(Ljava/lang/Object;)Z");
  return env->CallBooleanMethod(params, contains_key_method,
                                env->NewStringUTF(key));
}

#define RESOLVE_PARAM(key, cpp_type, java_type)                                \
  if (has_map_key(env, jParams, #key)) {                                       \
    params.key = get_map_value<java_type>(env, jParams, #key);                 \
  }

extern "C" JNIEXPORT jlong JNICALL Java_com_barkrn_BarkContext_nativeInitContext(
    JNIEnv *env, jclass type, jstring jPath, jobject jParams) {
  auto params = bark_context_default_params();
  RESOLVE_PARAM(verbosity, bark_verbosity_level, jint);
  RESOLVE_PARAM(temp, float, jfloat);
  RESOLVE_PARAM(fine_temp, float, jfloat);
  RESOLVE_PARAM(min_eos_p, float, jfloat);
  RESOLVE_PARAM(sliding_window_size, int, jint);
  RESOLVE_PARAM(max_coarse_history, int, jint);
  RESOLVE_PARAM(sample_rate, int, jint);
  RESOLVE_PARAM(target_bandwidth, int, jint);
  RESOLVE_PARAM(cls_token_id, int, jint);
  RESOLVE_PARAM(sep_token_id, int, jint);
  RESOLVE_PARAM(n_steps_text_encoder, int, jint);
  RESOLVE_PARAM(text_pad_token, int, jint);
  RESOLVE_PARAM(text_encoding_offset, int, jint);
  RESOLVE_PARAM(semantic_rate_hz, float, jfloat);
  RESOLVE_PARAM(semantic_pad_token, int, jint);
  RESOLVE_PARAM(semantic_vocab_size, int, jint);
  RESOLVE_PARAM(semantic_infer_token, int, jint);
  RESOLVE_PARAM(coarse_rate_hz, float, jfloat);
  RESOLVE_PARAM(coarse_infer_token, int, jint);
  RESOLVE_PARAM(coarse_semantic_pad_token, int, jint);
  RESOLVE_PARAM(n_coarse_codebooks, int, jint);
  RESOLVE_PARAM(n_fine_codebooks, int, jint);
  RESOLVE_PARAM(codebook_size, int, jint);
  int seed = 0;
  if (has_map_key(env, jParams, "seed")) {
    seed = get_map_value<jint>(env, jParams, "seed");
  }
  auto model_path = env->GetStringUTFChars(jPath, nullptr);
  bark_context *context = bark_load_model(model_path, params, seed);
  env->ReleaseStringUTFChars(jPath, model_path);
  return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_barkrn_BarkContext_nativeGenerate(
    JNIEnv *env, jclass type, jlong jCtx, jstring jText, jstring jOutPath,
    jint jThreads) {
  auto context = reinterpret_cast<bark_context *>(jCtx);
  int threads = jThreads;
  if (threads < 0) {
    threads = std::thread::hardware_concurrency() << 1;
  }
  if (threads <= 0) {
    threads = 1;
  }
  auto text = env->GetStringUTFChars(jText, nullptr);
  auto success = bark_generate_audio(context, text, threads);
  env->ReleaseStringUTFChars(jText, text);
  const float *audio_data = bark_get_audio_data(context);
  const int audio_samples = bark_get_audio_data_size(context);
  const auto sample_rate = context->params.sample_rate;
  if (success) {
    auto dest_path = env->GetStringUTFChars(jOutPath, nullptr);
    std::vector<float> audio_data_vec(audio_data, audio_data + audio_samples);
    pcmToWav(audio_data_vec, sample_rate, dest_path);
    env->ReleaseStringUTFChars(jOutPath, dest_path);
  }
  const auto load_time = bark_get_load_time(context);
  const auto eval_time = bark_get_eval_time(context);
  auto result_class = env->FindClass("com/barkrn/BarkContext$BarkResult");
  jobject result = env->NewObject(
      result_class, env->GetMethodID(result_class, "<init>", "(ZII)V"), success,
      load_time, eval_time);
  bark_reset_statistics(context);
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_barkrn_BarkContext_nativeReleaseContext(JNIEnv *env, jclass type,
                                                 jlong context) {
  bark_free(reinterpret_cast<bark_context *>(context));
}
