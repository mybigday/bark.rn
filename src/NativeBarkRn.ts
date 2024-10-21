import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export enum BarkVerbosityLevel {
  LOW = 0,
  MEDIUM = 1,
  HIGH = 2,
}

export interface BarkContextParams {
  seed?: number;
  verbosity?: BarkVerbosityLevel;
  temp?: number;
  fine_temp?: number;
  min_eos_p?: number;
  sliding_window_size?: number;
  max_coarse_history?: number;
  sample_rate?: number;
  target_bandwidth?: number;
  cls_token_id?: number;
  sep_token_id?: number;
  n_steps_text_encoder?: number;
  text_pad_token?: number;
  text_encoding_offset?: number;
  semantic_rate_hz?: number;
  semantic_pad_token?: number;
  semantic_vocab_size?: number;
  semantic_infer_token?: number;
  coarse_rate_hz?: number;
  coarse_infer_token?: number;
  coarse_semantic_pad_token?: number;
  n_coarse_codebooks?: number;
  n_fine_codebooks?: number;
  codebook_size?: number;
}

export interface Spec extends TurboModule {
  init_context(model_path: string, params: BarkContextParams): Promise<number>;
  generate(
    id: number,
    text: string,
    audio_path: string,
    threads: number
  ): Promise<{ success: boolean; load_time: number; eval_time: number }>;
  release_context(id: number): Promise<void>;
  release_all_contexts(): Promise<void>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('BarkRn');
