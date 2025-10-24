#include <iostream>
#include <string>
#include <iomanip>
#include <cstdlib>

// Include our new library headers
#include "fft_types.h"
#include "fft_lib.h"
#include "fft_utils.h" // For dft_naive and checkError
const bool DEBUG_VERBOSE = true;

/**
 * @brief Runs a single FFT test case.
 */
void runTest(const std::string &testName, const std::vector<int> &radices)
{
    std::cout << "=======================================\n";
    std::cout << "Test Case: " << testName << "\n";
    std::cout << "Radices: { ";
    for (int r : radices)
        std::cout << r << " ";
    std::cout << "}\n";

    int N = 1;
    for (int r : radices)
        N *= r;

    // 1. Create a test signal
    CVector x_fft(N);
    for (int i = 0; i < N; ++i)
    {
        x_fft[i] = Complex(std::rand() / (double)RAND_MAX, std::rand() / (double)RAND_MAX);
    }
    CVector x_naive = x_fft; // Keep a copy for the naive DFT

    if (DEBUG_VERBOSE)
    {
        std::cout << "Input Signal:\n";
        for (int i = 0; i < N; ++i)
        {
            std::cout << "x[" << i << "] = " << x_fft[i] << "\n";
        }
    }

    // 2. Run the Mixed-Radix FFT
    mixedRadixFFT(x_fft, radices);

    // 3. Run the Naive DFT
    CVector x_dft = dft_naive(x_naive);

    // 4. Compare results
    double error = checkError(x_fft, x_dft);
    std::cout << "N = " << N << ", Error vs. Naive DFT: " << error << "\n";
    std::cout << (error < 1e-9 ? "--- PASSED ---" : "--- FAILED ---") << "\n";

    // Optional: Print small vector results
    if (N <= 12)
    {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << " k |     FFT Output     |     Naive DFT    \n";
        std::cout << "---|--------------------|------------------\n";
        for (int i = 0; i < N; ++i)
        {
            std::cout << std::setw(2) << i << " | "
                      << std::setw(8) << x_fft[i].real() << " "
                      << std::setw(8) << x_fft[i].imag() << "j | "
                      << std::setw(8) << x_dft[i].real() << " "
                      << std::setw(8) << x_dft[i].imag() << "j\n";
        }
    }
}

int main()
{
    std::srand(42); // Seed for reproducible results

    // runTest("Radix-2 (N=8)", {2, 2, 2});
    // runTest("Mixed-Radix (N=6)", {2, 3});
    // runTest("Mixed-Radix (N=12)", {2, 2, 3});
    // runTest("Mixed-Radix (N=15)", {3, 5});
    // runTest("Radix-3 (N=9)", {3, 3});
    // runTest("Mixed-Radix (N=18)", {2, 3, 3});
    // runTest("Mixed-Radix (N=30)", {2, 3, 5});
    runTest("Mixed-Radix (N=60)", {2, 2, 3, 5});

    return 0;
}