
#import "BarkRn.h"
#import "Convert.h"
#import "BarkContext.h"

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

- (void)dealloc {
    [self.contexts removeAllObjects];
    [super dealloc];
}

RCT_EXPORT_METHOD(init_context: (NSString *)model_path
                  params: (NSDictionary *)ns_params
                  resolve: (RCTPromiseResolveBlock)resolve
                  reject: (RCTPromiseRejectBlock)reject) {
    try {
        BarkContext *context = [[BarkContext alloc] initWithModelPath:model_path params:ns_params];
        self.contexts[self.next_id] = context;
        resolve(self.next_id);
        self.next_id++;
    } @catch (NSException *e) {
        reject(@"init_context", @"Failed to init context", e);
    }
}

RCT_EXPORT_METHOD(generate: (NSInteger)_id
                  text: (NSString *)text
                  out_path: (NSString *)out_path
                  resolve: (RCTPromiseResolveBlock)resolve
                  reject: (RCTPromiseRejectBlock)reject) {
    if (!self.contexts[_id]) {
        reject(@"generate", @"Context not found", nil);
        return;
    }
    BarkContext *context = self.contexts[_id];
    NSDictionary *result = [context generate:text out_path:out_path];
    resolve(result);
}

RCT_EXPORT_METHOD(release_context: (NSInteger)_id
                  resolve: (RCTPromiseResolveBlock)resolve
                  reject: (RCTPromiseRejectBlock)reject) {
    if (self.contexts[_id]) {
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
