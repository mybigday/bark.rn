import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  init_context(
    model_path: string,
    params: Record<string, any>
  ): Promise<number>;
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
