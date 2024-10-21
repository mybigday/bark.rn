#import "BarkContext.h"
#import "Convert.h"

#include "utils.h"
#include <vector>
#include <thread>

@interface BarkContext ()

@property (nonatomic, assign) bark_context *context;
@property (nonatomic, assign) int sample_rate;
@property (nonatomic, assign) int n_threads;

@end

@implementation BarkContext

+ (BarkContext *)initWithModelPath:(NSString *)model_path params:(NSDictionary *)ns_params {
    self = [super init];
    if (self) {
        int seed = 0;
        if (ns_params && ns_params[@"seed"]) seed = [ns_params[@"seed"] intValue];
        self.sample_rate = 24000;
        if (ns_params && ns_params[@"sample_rate"]) self.sample_rate = [ns_params[@"sample_rate"] intValue];
        self.n_threads = -1;
        if (ns_params && ns_params[@"n_threads"]) self.n_threads = [ns_params[@"n_threads"] intValue];
        if (self.n_threads < 0) self.n_threads = std::thread::hardware_concurrency() << 1;
        if (self.n_threads == 0) self.n_threads = 1;
        bark_context_params params = [Convert convert_params:ns_params];
        try {
            self.context = bark_load_model(model_path, params, seed);
        } catch (const std::exception &e) {
            @throw [NSException exceptionWithName:@"BarkContext" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
        }
    }
    return self;
}

- (void)dealloc {
    bark_free(self.context);
    self.context = NULL;
    [super dealloc];
}

- (NSDictionary *)generate:(NSString *)text out_path:(NSString *)out_path {
    try {
        bool success = bark_generate_audio(self.context, text, self.n_threads);
    } catch (const std::exception &e) {
        @throw [NSException exceptionWithName:@"BarkContext" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    }
    if (success) {
        int audio_samples = bark_get_audio_data_size(self.context);
        const float *audio_data = bark_get_audio_data(self.context);
        std::vector<float> audio_data_vec(audio_data, audio_data + audio_samples);
        barkrn::pcmToWav(audio_data_vec, self.sample_rate, out_path);
    }
    return @{
        @"success": @(success),
        @"load_time": @(bark_get_load_time(self.context)),
        @"eval_time": @(bark_get_eval_time(self.context))
    };
}

@end
