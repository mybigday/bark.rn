#define DR_WAV_IMPLEMENTATION
#include "utils.h"
#include "dr_wav.h"

namespace barkrn {

void pcmToWav(float *data, int size, const int sample_rate,
              const char *dest_path) {
  drwav_data_format format;
  format.bitsPerSample = 32;
  format.sampleRate = sample_rate;
  format.container = drwav_container_riff;
  format.channels = 1;
  format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

  drwav wav;
  drwav_init_file_write(&wav, dest_path, &format, NULL);
  drwav_uint64 frames = drwav_write_pcm_frames(&wav, size, data);
  drwav_uninit(&wav);
}

} // namespace barkrn
