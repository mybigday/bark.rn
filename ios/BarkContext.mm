#import "BarkContext.h"
#import "Convert.h"

#include "bark.h"
#include "utils.h"
#include <vector>
#include <thread>
#include <memory>

@implementation BarkContext {
    std::shared_ptr<bark_context> context;
    int n_threads;
    int sample_rate;
}

- (instancetype)initWithModelPath:(NSString *)model_path params:(NSDictionary *)ns_params {
    self = [super init];
    if (self) {
        int seed = 0;
        if (ns_params && ns_params[@"seed"]) seed = [ns_params[@"seed"] intValue];
        sample_rate = 24000;
        if (ns_params && ns_params[@"sample_rate"]) sample_rate = [ns_params[@"sample_rate"] intValue];
        n_threads = -1;
        if (ns_params && ns_params[@"n_threads"]) n_threads = [ns_params[@"n_threads"] intValue];
        if (n_threads < 0) n_threads = std::thread::hardware_concurrency() << 1;
        if (n_threads == 0) n_threads = 1;
        bark_context_params params = [Convert convert_params:ns_params];
        try {
            bark_context *ctx = bark_load_model([model_path UTF8String], params, seed);
            if (ctx == nullptr) {
                @throw [NSException exceptionWithName:@"BarkContext" reason:@"Failed to load model" userInfo:nil];
            }
            context = std::shared_ptr<bark_context>(ctx, bark_free);
        } catch (const std::exception &e) {
            @throw [NSException exceptionWithName:@"BarkContext" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
        }
    }
    return self;
}

- (NSDictionary *)generate:(NSString *)text out_path:(NSString *)out_path {
    if (context == nullptr) {
        @throw [NSException exceptionWithName:@"BarkContext" reason:@"Context not initialized" userInfo:nil];
    }
    bool success = false;
    try {
        success = bark_generate_audio(context.get(), [text UTF8String], n_threads);
    } catch (const std::exception &e) {
        @throw [NSException exceptionWithName:@"BarkContext" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    }
    if (success) {
        int audio_samples = bark_get_audio_data_size(context.get());
        const float *audio_data = bark_get_audio_data(context.get());
        std::vector<float> audio_data_vec(audio_data, audio_data + audio_samples);
        barkrn::pcmToWav(audio_data_vec, sample_rate, [out_path UTF8String]);
    }
    return @{
        @"success": @(success),
        @"load_time": @(bark_get_load_time(context.get())),
        @"eval_time": @(bark_get_eval_time(context.get()))
    };
}

@end
