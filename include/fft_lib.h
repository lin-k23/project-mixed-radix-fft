#pragma once
#include "fft_types.h"
#include <vector>

/**
 * @brief Performs an in-place mixed-radix DIT FFT.
 *
 * @param data The vector of complex data. Reordered and transformed in-place.
 * @param radices The vector of factors {r1, r2, ..., rm}.
 */
void mixedRadixFFT(CVector &data, const std::vector<int> &radices);