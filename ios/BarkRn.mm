
#import "BarkRn.h"
#import "Convert.h"

#include <thread>

@interface BarkRn ()

@property (nonatomic, retain) NSMutableDictionary *contexts;
@property (nonatomic, assign) NSInteger next_id;

@end

@implementation BarkRn
RCT_EXPORT_MODULE()


- (instancetype)init {
    self = [super init];
    if (self) {
        self.contexts = [NSMutableDictionary dictionary];
        self.next_id = 0;
    }
    return self;
}

RCT_EXPORT_METHOD(init_context: (NSString *)model_path
                  params: (NSDictionary *)ns_params
                  resolve: (RCTPromiseResolveBlock)resolve
                  reject: (RCTPromiseRejectBlock)reject) {
    try {
        bark_context_params params = [Convert convert_params:ns_params];
        int seed = 0;
        if (ns_params && ns_params[@"seed"]) seed = [ns_params[@"seed"] intValue];
        self.contexts[self.next_id] = bark_load_model(model_path, params, seed);
        resolve(self.next_id);
        self.next_id++;
    } catch (const std::exception &e) {
        reject(@"init_context", @"Failed to init context", e);
    } @catch (NSException *e) {
        reject(@"init_context", @"Failed to init context", e);
    }
}

RCT_EXPORT_METHOD(generate: (NSInteger)_id
                  text: (NSString *)text
                  out_path: (NSString *)out_path
                  threads: (NSInteger)threads
                  resolve: (RCTPromiseResolveBlock)resolve
                  reject: (RCTPromiseRejectBlock)reject) {
    if (!self.contexts[_id]) {
        reject(@"generate", @"Context not found", nil);
        return;
    }
    int _threads = [threads intValue];
    if (_threads < 0) {
        _threads = std::thread::hardware_concurrency() << 1;
    }
    if (_threads <= 0) {
        _threads = 1;
    }
    bark_context *context = self.contexts[_id];
    bool success = bark_generate_audio(context, text, _threads);
    if (success) {
        int sample_rate = context->params.sample_rate;
        int audio_samples = bark_get_audio_data_size(context);
        const float *audio_data = bark_get_audio_data(context);
        std::vector<float> audio_data_vec(audio_data, audio_data + audio_samples);
        pcmToWav(audio_data_vec, sample_rate, out_path);
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    result[@"success"] = @(success);
    result[@"load_time"] = @(bark_get_load_time(context));
    result[@"eval_time"] = @(bark_get_eval_time(context));
    resolve(result);
}

RCT_EXPORT_METHOD(release_context: (NSInteger)_id
                  resolve: (RCTPromiseResolveBlock)resolve
                  reject: (RCTPromiseRejectBlock)reject) {
    if (self.contexts[_id]) {
        bark_free(self.contexts[_id]);
        [self.contexts removeObjectForKey:@(_id)];
    }
    resolve(nil);
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeBarkRnSpecJSI>(params);
}
#endif

@end
