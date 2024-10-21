#ifndef BARKRN_H
#define BARKRN_H
#include <string>
#include <vector>

namespace barkrn {
void pcmToWav(const std::vector<float> &data, const int sample_rate,
              const std::string dest_path);
}

#endif /* BARKRN_H */
