#include "fft_utils.h"
#include <stdexcept> // For std::runtime_error
#include <vector>    // For std::vector<bool>

// =================================================================
// PHASE 1: DIGIT-REVERSAL PERMUTATION (DIT Version)
// =================================================================

/**
 * @brief Calculates the digit-reversed index for a DIT FFT.
 *
 * An index n = n_0 + n_1*r1 + n_2*r1*r2 + ...
 * is mapped to:
 * j = n_0*(N/r1) + n_1*(N/(r1*r2)) + n_2*(N/(r1*r2*r3)) + ...
 *
 * @param n The input index (0 to N-1).
 * @param radices The vector of factors {r1, r2, ..., rm}.
 * @param N The total size, N.
 * @return The permuted (DIT digit-reversed) index j.
 */
int getDigitReversedIndex(int n, const std::vector<int> &radices, int N)
{
    int m = radices.size();
    int j = 0;
    int n_remaining = n;
    int n_blocksize = 1;

    for (int i = 0; i < m; ++i)
    {
        int r_i = radices[i];
        int digit = n_remaining % r_i;
        n_remaining /= r_i;

        j += digit * (N / (n_blocksize * r_i));
        n_blocksize *= r_i;
    }
    return j;
}

/**
 * @brief Generates the full DIT permutation table.
 */
std::vector<int> generatePermutationTable(int N, const std::vector<int> &radices)
{
    std::vector<int> table(N);
    for (int n = 0; n < N; ++n)
    {
        table[n] = getDigitReversedIndex(n, radices, N);
    }
    return table;
}

/**
 * @brief Reorders a vector of data in-place using the permutation table.
 *
 * data_new[n]=data_old[table[n]]
 */
void reorderInPlace(CVector &data, const std::vector<int> &table)
{
    if (data.size() != table.size())
    {
        throw std::runtime_error("Data and table sizes do not match.");
    }
    int N = data.size();
    std::vector<bool> moved(N, false); // Track which items are in final place

    for (int n = 0; n < N; ++n)
    {
        if (moved[n])
        {
            continue; // This element was part of a cycle we already processed
        }

        if (table[n] == n)
        {
            moved[n] = true; // Handle 1-cycles (no move)
            continue;
        }

        // We are at the start of a new cycle
        int current = n;
        Complex temp = data[n]; // Store the value from the start of the cycle

        while (true)
        {
            int next = table[current];
            moved[current] = true;

            if (next == n)
            {
                // We've completed the cycle.
                // Place the stored 'temp' value at the *end* of the cycle.
                //
                // THIS IS THE FIX:
                // data[n] = temp;      <-- BUGGY LINE
                data[current] = temp; // <-- CORRECT LINE
                break;
            }

            // Shift the value from the next position to the current position
            data[current] = data[next];
            current = next; // Move to the next element in the cycle
        }
    }
}

// =================================================================
// TESTBENCH
// =================================================================

/**
 * @brief A simple O(N^2) DFT for verification.
 */
CVector dft_naive(const CVector &x)
{
    int N = x.size();
    CVector X(N);
    for (int k = 0; k < N; ++k)
    {
        X[k] = 0.0;
        for (int n = 0; n < N; ++n)
        {
            double angle = -2.0 * PI * (double)(k * n) / (double)N;
            X[k] += x[n] * std::exp(Complex(0, angle));
        }
    }
    return X;
}

/**
 * @brief Calculates the L2-norm of the error between two vectors.
 */
double checkError(const CVector &a, const CVector &b)
{
    double error = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        error += std::norm(a[i] - b[i]); // std::norm is |z|^2
    }
    return std::sqrt(error);
}

std::vector<int> invertPermutationTable(const std::vector<int> &table)
{
    int N = table.size();
    std::vector<int> inv_table(N);
    for (int n = 0; n < N; ++n)
    {
        inv_table[table[n]] = n;
    }
    return inv_table;
}
