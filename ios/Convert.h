#ifdef __cplusplus
#import "bark.h"
#endif

@interface Convert : NSObject

+ (bark_context_params)convert_params:(NSDictionary *)ns_params;

@end
