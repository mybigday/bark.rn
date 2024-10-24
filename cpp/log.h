#pragma once

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "BarkRN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#elif __APPLE__
#define LOGI(...) printf("[BarkRN] INFO: "); printf(__VA_ARGS__); printf("\n")
#define LOGD(...) printf("[BarkRN] DEBUG: "); printf(__VA_ARGS__); printf("\n")
#define LOGW(...) printf("[BarkRN] WARN: "); printf(__VA_ARGS__); printf("\n")
#define LOGE(...) printf("[BarkRN] ERROR: "); printf(__VA_ARGS__); printf("\n")
#else
#define LOGI(...) fprintf(stderr, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif
