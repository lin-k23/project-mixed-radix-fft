#include "fft_lib.h"
#include "fft_utils.h" // For permutation
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream> // added for debug
#include <cmath>    // for std::abs
#include <limits>   // for std::numeric_limits

// --- Type Aliases ---
// We get these from "fft_types.h"

#define DEBUG_MRFFT

/**
 * @brief Performs an in-place mixed-radix DIT FFT.
 */
void mixedRadixFFT(CVector &data, const std::vector<int> &radices)
{
    int N = data.size();
    int m = radices.size();

    // Verify N matches the product of radices
    int N_check = 1;
    for (int r : radices)
        N_check *= r;
    if (N != N_check)
    {
        throw std::invalid_argument("N does not match product of radices.");
    }

    // Save original input for diagnostics
#ifdef DEBUG_MRFFT
    std::vector<Complex> x_before_perm(data.begin(), data.end());
#endif

    // ------------------------------------
    // PHASE 1: Permutation
    // Use reversed radices for the permutation so it lines up with the DIT stage order
    // we apply below (we run stages using reversed radices).
    std::vector<int> rad_rev = radices;
    std::reverse(rad_rev.begin(), rad_rev.end());
    std::vector<int> scatter_table = generatePermutationTable(N, rad_rev);
    std::vector<int> gather_table = invertPermutationTable(scatter_table);
    // reorderInPlace expects a table mapping new_index -> old_index,
    // so use the inverted (gather) table here.
    reorderInPlace(data, gather_table);

#ifdef DEBUG_MRFFT
    // Print original and permuted input (helpful to detect permutation issues)
    auto print_vec = [&](const std::string &label, const CVector &v)
    {
        std::cerr << label << " (N=" << v.size() << "):\n";
        size_t limit = v.size() > 256 ? 256 : v.size();
        for (size_t i = 0; i < limit; ++i)
            std::cerr << " [" << i << "] = " << v[i] << "\n";
        if (v.size() > limit)
            std::cerr << " ... (truncated)\n";
    };
    print_vec("original input (saved)", CVector(x_before_perm.begin(), x_before_perm.end()));
    print_vec("after permutation (data)", data);
#endif

    // ------------------------------------
    // PHASE 2: Computation (m stages)
    // We'll run the stage loop via a small helper so we can try different radices orderings
    // for debugging (original vs reversed).
    auto run_stages = [&](CVector &work, const std::vector<int> &rads, const std::string &label, std::vector<CVector> *snapshots = nullptr)
    {
        int span_local = 1;
        int m_local = (int)rads.size();
        for (int s_local = 0; s_local < m_local; ++s_local)
        {
            int r = rads[s_local];
            int block_size = span_local * r;
            CVector temp(r);
            CVector dft_out(r);

            for (int b = 0; b < N / block_size; ++b)
            {
                for (int k = 0; k < span_local; ++k)
                {
                    // Load
                    for (int j = 0; j < r; ++j)
                        temp[j] = work[b * block_size + k + j * span_local];

                    // local r-point DFT
                    if (r == 2)
                    {
                        dft_out[0] = temp[0] + temp[1];
                        dft_out[1] = temp[0] - temp[1];
                    }
                    else
                    {
                        for (int j_out = 0; j_out < r; ++j_out)
                        {
                            Complex sum = 0.0;
                            for (int j_in = 0; j_in < r; ++j_in)
                            {
                                double angle = -2.0 * PI * (double)(j_out * j_in) / (double)r;
                                Complex W_r = std::exp(Complex(0, angle));
                                sum += temp[j_in] * W_r;
                            }
                            dft_out[j_out] = sum;
                        }
                    }

#ifdef DEBUG_MRFFT
                    std::cerr << "[DEBUG_MRFFT:" << label << "] stage=" << s_local << " block=" << b << " k=" << k << " r=" << r << "\n";
                    for (int jj = 0; jj < r; ++jj)
                        std::cerr << "  temp[" << jj << "] = " << temp[jj] << "\n";
                    for (int jj = 0; jj < r; ++jj)
                        std::cerr << "  dft_out[" << jj << "] = " << dft_out[jj] << "\n";
#endif

                    for (int j = 0; j < r; ++j)
                    {
                        // Use full transform length N in denominator
                        double angle = -2.0 * PI * (double)(j * (b * span_local + k)) / (double)N;
                        Complex twiddle = std::exp(Complex(0, angle));
#ifdef DEBUG_MRFFT
                        std::cerr << "    j=" << j << " twiddle(angle)=" << angle << " twiddle=" << twiddle << "\n";
                        std::cerr << "    write idx=" << (b * block_size + k + j * span_local) << " = " << dft_out[j] * twiddle << "\n";
#endif
                        work[b * block_size + k + j * span_local] = dft_out[j] * twiddle;
                    }
                }
            }

#ifdef DEBUG_MRFFT
            std::cerr << "[DEBUG_MRFFT:" << label << "] After stage " << s_local << " (r=" << r << ", span=" << span_local << "):\n";
            size_t limit = work.size() > 256 ? 256 : work.size();
            for (size_t i = 0; i < limit; ++i)
                std::cerr << " [" << i << "] = " << work[i] << "\n";
            if (work.size() > limit)
                std::cerr << " ... (truncated)\n";
            if (snapshots)
                snapshots->push_back(work);
#endif

            span_local = block_size;
        }
    };

    // Helper to compute expected snapshots for N=6 using math (twiddle denom = N)
    auto compute_expected_snapshots_N6 = [&](const CVector &perm_data, const std::vector<int> &rads_rev)
    {
        const int Nlocal = 6;
        std::vector<CVector> expect;
        // Stage 0
        CVector s0 = perm_data;
        int r0 = rads_rev[0];
        int span0 = 1;
        int block0 = span0 * r0;
        for (int b = 0; b < Nlocal / block0; ++b)
        {
            for (int k = 0; k < span0; ++k)
            {
                CVector temp(r0);
                for (int j = 0; j < r0; ++j)
                    temp[j] = s0[b * block0 + k + j * span0];
                for (int j_out = 0; j_out < r0; ++j_out)
                {
                    Complex sum = 0.0;
                    for (int j_in = 0; j_in < r0; ++j_in)
                    {
                        double angle = -2.0 * PI * (double)(j_out * j_in) / (double)r0;
                        sum += temp[j_in] * std::exp(Complex(0, angle));
                    }
                    s0[b * block0 + k + j_out * span0] = sum;
                }
            }
        }
        expect.push_back(s0);

        // Stage 1
        CVector s1 = s0;
        int r1 = rads_rev[1];
        int span1 = block0;
        int block1 = span1 * r1;
        for (int b = 0; b < Nlocal / block1; ++b)
        {
            for (int k = 0; k < span1; ++k)
            {
                CVector temp(r1);
                for (int j = 0; j < r1; ++j)
                    temp[j] = s1[b * block1 + k + j * span1];
                for (int j_out = 0; j_out < r1; ++j_out)
                {
                    Complex sum = 0.0;
                    for (int j_in = 0; j_in < r1; ++j_in)
                    {
                        double angle = -2.0 * PI * (double)(j_out * j_in) / (double)r1;
                        sum += temp[j_in] * std::exp(Complex(0, angle));
                    }
                    double tw_angle = -2.0 * PI * (double)(j_out * (b * span1 + k)) / (double)Nlocal;
                    Complex tw = std::exp(Complex(0, tw_angle));
                    s1[b * block1 + k + j_out * span1] = sum * tw;
                }
            }
        }
        expect.push_back(s1);
        return expect;
    };

    // The correct DIT mixed-radix sequence is: permute, then run stages with radices
    // in reversed order relative to the digit-reversal construction. Use reversed radices.
    // rad_rev was computed above for the permutation; reuse it here.
    std::vector<CVector> snapshots;
    if (N == 6)
    {
        // copy permuted input for expected calculation
        CVector permuted_input = data;
        run_stages(data, rad_rev, "rev", &snapshots);
        auto expected = compute_expected_snapshots_N6(permuted_input, rad_rev);
        for (size_t s = 0; s < expected.size(); ++s)
        {
            std::cerr << "[DEBUG_MRFFT] Stage " << s << " comparison:\n";
            for (int i = 0; i < N; ++i)
            {
                Complex actual = snapshots[s][i];
                Complex expv = expected[s][i];
                double err = std::abs(actual - expv);
                if (err > 1e-9)
                    std::cerr << " idx=" << i << " actual=" << actual << " expected=" << expv << " err=" << err << "\n";
            }
        }
    }
    else
    {
        run_stages(data, rad_rev, "rev");
    }

#ifdef DEBUG_MRFFT
    // Diagnostic: compute naive DFT of the original *input* (before permutation)
    // To compare correctly, we need the original input. Since we permuted in-place at the start,
    // regenerate the original ordering by applying the scatter_table (inverse of gather_table).
    // scatter_table maps old_index -> new_index (original -> permuted). We saved scatter_table earlier,
    // so reconstruct original input by undoing reorderInPlace: create copy and scatter back.

    // Use the original saved input (x_before_perm) to compute the reference DFT
    // (x_before_perm was saved at the start of this function before any permutation).
    CVector x_orig = CVector(x_before_perm.begin(), x_before_perm.end());
    std::vector<Complex> X_ref = dft_naive(x_orig);

    // Our algorithm produced data in-place; after full mixedRadixFFT the 'data' vector
    // should equal X_ref (up to numerical error) but note: depending on permutation convention,
    // the output ordering might differ. We check for equality in the straightforward output order.
    double max_err = 0.0;
    int bad_idx = -1;
    for (int i = 0; i < N; ++i)
    {
        double e = std::abs(data[i] - X_ref[i]);
        if (e > max_err)
        {
            max_err = e;
            bad_idx = i;
        }
    }

    if (max_err > 1e-9)
    {
        std::cerr << "[DEBUG_MRFFT] mismatch: max_err=" << max_err << " at index " << bad_idx << "\n";
        std::cerr << " data[" << bad_idx << "] = " << data[bad_idx] << "\n";
        std::cerr << " ref [" << bad_idx << "] = " << X_ref[bad_idx] << "\n";
    }
    else
    {
        std::cerr << "[DEBUG_MRFFT] OK: max_err=" << max_err << "\n";
    }

    // Additional diagnostics: compute DFT of the permuted input (x_perm_diag)
    // and compute nearest-neighbor mapping from computed outputs to naive DFT indices.
    std::vector<Complex> x_perm_diag(N);
    // scatter_table maps original_index -> permuted_index
    for (int orig = 0; orig < N; ++orig)
    {
        int perm_i = scatter_table[orig];
        x_perm_diag[perm_i] = x_before_perm[orig];
    }
    std::vector<Complex> X_perm_ref = dft_naive(CVector(x_perm_diag.begin(), x_perm_diag.end()));
    std::cerr << "[DEBUG_MRFFT] Naive DFT of permuted input (x_perm) :\n";
    for (int i = 0; i < N; ++i)
        std::cerr << "  X_perm_ref[" << i << "] = " << X_perm_ref[i] << "\n";

    // For each computed output element, find the closest entry in X_ref (naive) and report
    std::cerr << "[DEBUG_MRFFT] nearest-neighbor mapping (computed_index -> naive_index, error):\n";
    for (int i = 0; i < N; ++i)
    {
        double best_err = std::numeric_limits<double>::infinity();
        int best_j = -1;
        for (int j = 0; j < N; ++j)
        {
            double e = std::abs(data[i] - X_ref[j]);
            if (e < best_err)
            {
                best_err = e;
                best_j = j;
            }
        }
        std::cerr << "  computed[" << i << "] best matches naive index " << best_j << " with err=" << best_err << "\n";
    }

    std::cerr << "[DEBUG_MRFFT] nearest-neighbor mapping (computed -> permuted-naive):\n";
    for (int i = 0; i < N; ++i)
    {
        double best_err = std::numeric_limits<double>::infinity();
        int best_j = -1;
        for (int j = 0; j < N; ++j)
        {
            double e = std::abs(data[i] - X_perm_ref[j]);
            if (e < best_err)
            {
                best_err = e;
                best_j = j;
            }
        }
        std::cerr << "  computed[" << i << "] best matches permuted-naive index " << best_j << " with err=" << best_err << "\n";
    }
#endif
}