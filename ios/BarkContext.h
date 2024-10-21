@interface BarkContext : NSObject

- (instancetype)initWithModelPath:(NSString *)model_path params:(NSDictionary *)ns_params;
- (NSDictionary *)generate:(NSString *)text out_path:(NSString *)out_path;

@end
