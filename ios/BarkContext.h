#ifdef __cplusplus
#import "bark.h"
#endif

@interface BarkContext : NSObject

+ (BarkContext *)initWithModelPath:(NSString *)model_path params:(NSDictionary *)ns_params;
- (void)dealloc;
- (NSDictionary *)generate:(NSString *)text out_path:(NSString *)out_path threads:(NSInteger)threads sample_rate:(NSInteger)sample_rate;

@end
