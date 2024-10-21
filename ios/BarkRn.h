#ifdef RCT_NEW_ARCH_ENABLED
#import "RNBarkRnSpec.h"

@interface BarkRn : NSObject <NativeBarkRnSpec>
#else
#import <React/RCTBridgeModule.h>

@interface BarkRn : NSObject <RCTBridgeModule>
#endif

@end
