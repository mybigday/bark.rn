#include <cstddef>

#ifndef BARKRN_H
#define BARKRN_H

namespace barkrn {
void pcmToWav(float *data, int size, const int sample_rate,
              const char *dest_path);
}

#endif /* BARKRN_H */
