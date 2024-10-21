#include "utils.h"
#include "bark.h"
#include <jni.h>
#include <thread>
#include <tuple>
#include <type_traits>

template <typename T>
T get_map_value(JNIEnv *env, jobject params, const char *key) {
  jclass map_class = env->FindClass("java/util/Map");
  jmethodID get_method = env->GetMethodID(
      map_class, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
  jobject value = env->CallObjectMethod(params, get_method, env->NewStringUTF(key));
  if constexpr (std::is_same_v<T, jfloat>) {
    jclass float_class = env->FindClass("java/lang/Float");
    return env->CallFloatMethod(value, env->GetMethodID(float_class, "floatValue", "()F"));
  } else if constexpr (std::is_same_v<T, jint>) {
    jclass int_class = env->FindClass("java/lang/Integer");
    return env->CallIntMethod(value, env->GetMethodID(int_class, "intValue", "()I"));
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

bool has_map_key(JNIEnv *env, jobject params, const char *key) {
  jclass map_class = env->FindClass("java/util/Map");
  jmethodID contains_key_method =
      env->GetMethodID(map_class, "containsKey", "(Ljava/lang/Object;)Z");
  return env->CallBooleanMethod(params, contains_key_method,
                                env->NewStringUTF(key));
}

extern "C" JNIEXPORT jlong JNICALL Java_com_barkrn_BarkContext_nativeInitContext(
    JNIEnv *env, jclass type, jstring jPath, jobject jParams) {
  auto params = bark_context_default_params();
  if (has_map_key(env, jParams, "verbosity")) {
    params.verbosity = static_cast<bark_verbosity_level>(get_map_value<jint>(env, jParams, "verbosity"));
  }
  if (has_map_key(env, jParams, "temp")) {
    params.temp = get_map_value<jfloat>(env, jParams, "temp");
  }
  if (has_map_key(env, jParams, "fine_temp")) {
    params.fine_temp = get_map_value<jfloat>(env, jParams, "fine_temp");
  }
  if (has_map_key(env, jParams, "min_eos_p")) {
    params.min_eos_p = get_map_value<jfloat>(env, jParams, "min_eos_p");
  }
  if (has_map_key(env, jParams, "sliding_window_size")) {
    params.sliding_window_size = get_map_value<jint>(env, jParams, "sliding_window_size");
  }
  if (has_map_key(env, jParams, "max_coarse_history")) {
    params.max_coarse_history = get_map_value<jint>(env, jParams, "max_coarse_history");
  }
  if (has_map_key(env, jParams, "sample_rate")) {
    params.sample_rate = get_map_value<jint>(env, jParams, "sample_rate");
  }
  if (has_map_key(env, jParams, "target_bandwidth")) {
    params.target_bandwidth = get_map_value<jint>(env, jParams, "target_bandwidth");
  }
  if (has_map_key(env, jParams, "cls_token_id")) {
    params.cls_token_id = get_map_value<jint>(env, jParams, "cls_token_id");
  }
  if (has_map_key(env, jParams, "sep_token_id")) {
    params.sep_token_id = get_map_value<jint>(env, jParams, "sep_token_id");
  }
  if (has_map_key(env, jParams, "n_steps_text_encoder")) {
    params.n_steps_text_encoder = get_map_value<jint>(env, jParams, "n_steps_text_encoder");
  }
  if (has_map_key(env, jParams, "text_pad_token")) {
    params.text_pad_token = get_map_value<jint>(env, jParams, "text_pad_token");
  }
  if (has_map_key(env, jParams, "text_encoding_offset")) {
    params.text_encoding_offset = get_map_value<jint>(env, jParams, "text_encoding_offset");
  }
  if (has_map_key(env, jParams, "semantic_rate_hz")) {
    params.semantic_rate_hz = get_map_value<jfloat>(env, jParams, "semantic_rate_hz");
  }
  if (has_map_key(env, jParams, "semantic_pad_token")) {
    params.semantic_pad_token = get_map_value<jint>(env, jParams, "semantic_pad_token");
  }
  if (has_map_key(env, jParams, "semantic_vocab_size")) {
    params.semantic_vocab_size = get_map_value<jint>(env, jParams, "semantic_vocab_size");
  }
  if (has_map_key(env, jParams, "semantic_infer_token")) {
    params.semantic_infer_token = get_map_value<jint>(env, jParams, "semantic_infer_token");
  }
  if (has_map_key(env, jParams, "coarse_rate_hz")) {
    params.coarse_rate_hz = get_map_value<jfloat>(env, jParams, "coarse_rate_hz");
  }
  if (has_map_key(env, jParams, "coarse_infer_token")) {
    params.coarse_infer_token = get_map_value<jint>(env, jParams, "coarse_infer_token");
  }
  if (has_map_key(env, jParams, "coarse_semantic_pad_token")) {
    params.coarse_semantic_pad_token = get_map_value<jint>(env, jParams, "coarse_semantic_pad_token");
  }
  if (has_map_key(env, jParams, "n_coarse_codebooks")) {
    params.n_coarse_codebooks = get_map_value<jint>(env, jParams, "n_coarse_codebooks");
  }
  if (has_map_key(env, jParams, "n_fine_codebooks")) {
    params.n_fine_codebooks = get_map_value<jint>(env, jParams, "n_fine_codebooks");
  }
  if (has_map_key(env, jParams, "codebook_size")) {
    params.codebook_size = get_map_value<jint>(env, jParams, "codebook_size");
  }
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
    jint jThreads, jint sample_rate) {
  auto context = reinterpret_cast<bark_context *>(jCtx);
  int threads = jThreads;
  if (threads < 0) {
    threads = std::thread::hardware_concurrency() << 1;
  } else if (threads == 0) {
    threads = 1;
  }
  auto text = env->GetStringUTFChars(jText, nullptr);
  auto success = bark_generate_audio(context, text, threads);
  env->ReleaseStringUTFChars(jText, text);
  const float *audio_data = bark_get_audio_data(context);
  const int audio_samples = bark_get_audio_data_size(context);
  if (success) {
    auto dest_path = env->GetStringUTFChars(jOutPath, nullptr);
    std::vector<float> audio_data_vec(audio_data, audio_data + audio_samples);
    barkrn::pcmToWav(audio_data_vec, sample_rate, dest_path);
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
