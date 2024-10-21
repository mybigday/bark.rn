import { NativeModules, Platform } from 'react-native';
import type { BarkContextParams } from './NativeBarkRn';

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

export type { BarkContextParams };

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
