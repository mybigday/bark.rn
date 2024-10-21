#import "Convert.h"

@implementation Convert

+ (bark_context_params)convert_params:(NSDictionary *)ns_params {
  bark_context_params params = bark_context_default_params();
  if (ns_params) {
    if (ns_params[@"verbosity"])
      params.verbosity =
          (bark_verbosity_level)[ns_params[@"verbosity"] intValue];
    if (ns_params[@"temp"])
      params.temp = [ns_params[@"temp"] floatValue];
    if (ns_params[@"fine_temp"])
      params.fine_temp = [ns_params[@"fine_temp"] floatValue];
    if (ns_params[@"min_eos_p"])
      params.min_eos_p = [ns_params[@"min_eos_p"] floatValue];
    if (ns_params[@"sliding_window_size"])
      params.sliding_window_size = [ns_params[@"sliding_window_size"] intValue];
    if (ns_params[@"max_coarse_history"])
      params.max_coarse_history = [ns_params[@"max_coarse_history"] intValue];
    if (ns_params[@"sample_rate"])
      params.sample_rate = [ns_params[@"sample_rate"] intValue];
    if (ns_params[@"target_bandwidth"])
      params.target_bandwidth = [ns_params[@"target_bandwidth"] intValue];
    if (ns_params[@"cls_token_id"])
      params.cls_token_id = [ns_params[@"cls_token_id"] intValue];
    if (ns_params[@"sep_token_id"])
      params.sep_token_id = [ns_params[@"sep_token_id"] intValue];
    if (ns_params[@"n_steps_text_encoder"])
      params.n_steps_text_encoder =
          [ns_params[@"n_steps_text_encoder"] intValue];
    if (ns_params[@"text_pad_token"])
      params.text_pad_token = [ns_params[@"text_pad_token"] intValue];
    if (ns_params[@"text_encoding_offset"])
      params.text_encoding_offset =
          [ns_params[@"text_encoding_offset"] intValue];
    if (ns_params[@"semantic_rate_hz"])
      params.semantic_rate_hz = [ns_params[@"semantic_rate_hz"] floatValue];
    if (ns_params[@"semantic_pad_token"])
      params.semantic_pad_token = [ns_params[@"semantic_pad_token"] intValue];
    if (ns_params[@"semantic_vocab_size"])
      params.semantic_vocab_size = [ns_params[@"semantic_vocab_size"] intValue];
    if (ns_params[@"semantic_infer_token"])
      params.semantic_infer_token =
          [ns_params[@"semantic_infer_token"] intValue];
    if (ns_params[@"coarse_rate_hz"])
      params.coarse_rate_hz = [ns_params[@"coarse_rate_hz"] floatValue];
    if (ns_params[@"coarse_infer_token"])
      params.coarse_infer_token = [ns_params[@"coarse_infer_token"] intValue];
    if (ns_params[@"coarse_semantic_pad_token"])
      params.coarse_semantic_pad_token =
          [ns_params[@"coarse_semantic_pad_token"] intValue];
    if (ns_params[@"n_coarse_codebooks"])
      params.n_coarse_codebooks = [ns_params[@"n_coarse_codebooks"] intValue];
    if (ns_params[@"n_fine_codebooks"])
      params.n_fine_codebooks = [ns_params[@"n_fine_codebooks"] intValue];
    if (ns_params[@"codebook_size"])
      params.codebook_size = [ns_params[@"codebook_size"] intValue];
  }
  return params;
}

@end
