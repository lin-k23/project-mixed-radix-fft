#pragma once
#include "fft_types.h"
#include <vector>

// --- Phase 1: Permutation ---
int getDigitReversedIndex(int n, const std::vector<int> &radices, int N);
std::vector<int> generatePermutationTable(int N, const std::vector<int> &radices);
void reorderInPlace(CVector &data, const std::vector<int> &table);

// --- Testbench Helpers ---
CVector dft_naive(const CVector &x);
double checkError(const CVector &a, const CVector &b);

std::vector<int> invertPermutationTable(const std::vector<int> &table);