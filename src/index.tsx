import { NativeModules, Platform } from 'react-native';

const LINKING_ERROR =
  `The package 'bark.rn' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo Go\n';

// @ts-expect-error
const isTurboModuleEnabled = global.__turboModuleProxy != null;

const BarkRnModule = isTurboModuleEnabled
  ? require('./NativeBarkRn').default
  : NativeModules.BarkRn;

const BarkRn = BarkRnModule
  ? BarkRnModule
  : new Proxy(
      {},
      {
        get() {
          throw new Error(LINKING_ERROR);
        },
      }
    );

export enum BarkVerbosityLevel {
  LOW = 0,
  MEDIUM = 1,
  HIGH = 2,
}

export type BarkContextParams = {
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
};

class BarkContext {
  id: number;

  constructor(id: number) {
    this.id = id;
  }

  static async load(
    model_path: string,
    params?: BarkContextParams
  ): Promise<BarkContext> {
    const id = await BarkRn.init_context(model_path, params ?? {});
    return new BarkContext(id);
  }

  generate(
    text: string,
    out_path: string,
    threads: number = -1
  ): Promise<{ success: boolean; load_time: number; eval_time: number }> {
    return BarkRn.generate(this.id, text, out_path, threads);
  }

  release(): Promise<void> {
    return BarkRn.release_context(this.id);
  }
}

export default BarkContext;
