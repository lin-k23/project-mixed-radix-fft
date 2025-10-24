#include "fft_utils.h"
#include "fft_types.h"
#include <iostream>
#include <vector>
#include <cassert> // For simple testing
#include <iomanip>
#include <cmath>

const bool DEBUG_VERBOSE = true;

// A simple helper to compare vectors
bool areVectorsEqual(const std::vector<int> &a, const std::vector<int> &b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

bool areComplexVectorsEqual(const CVector &a, const CVector &b, double tol = 1e-12)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i] - b[i]) > tol)
            return false;
    }
    return true;
}

void test_reorderInPlace_basicCycles()
{
    std::cout << "Running test: test_reorderInPlace_basicCycles..." << std::endl;

    // Case 1: simple 2-cycle (swap)
    {
        CVector data = {Complex(1.0, 0.0), Complex(2.0, 0.0)};
        std::vector<int> table = {1, 0}; // swap 0 <-> 1
        CVector expected(2);
        for (size_t n = 0; n < table.size(); ++n)
            expected[n] = data[table[n]];

        reorderInPlace(data, table);
        assert(areComplexVectorsEqual(data, expected));
        if (DEBUG_VERBOSE)
            std::cout << "  2-cycle PASSED\n";
    }

    // Case 2: single 4-cycle
    {
        CVector data = {Complex(0, 0), Complex(1, 0), Complex(2, 0), Complex(3, 0)};
        std::vector<int> table = {1, 2, 3, 0};
        CVector expected(4);
        for (size_t n = 0; n < table.size(); ++n)
            expected[n] = data[table[n]];

        reorderInPlace(data, table);
        assert(areComplexVectorsEqual(data, expected));
        if (DEBUG_VERBOSE)
            std::cout << "  4-cycle PASSED\n";
    }

    std::cout << "  basic cycles PASSED" << std::endl;
}

void test_reorderInPlace_withPermutationTable()
{
    std::cout << "Running test: test_reorderInPlace_withPermutationTable..." << std::endl;

    // Use the permutation table from the existing test: N=6, {2,3}
    std::vector<int> radices_6 = {2, 3};
    std::vector<int> table_6 = generatePermutationTable(6, radices_6);

    // Prepare distinct complex values so we can easily verify moves
    CVector data;
    for (int i = 0; i < 6; ++i)
        data.push_back(Complex((double)i + 0.1, (double)i + 0.2)); // distinct

    CVector expected(6);
    for (size_t n = 0; n < table_6.size(); ++n)
        expected[n] = data[table_6[n]];

    reorderInPlace(data, table_6);
    if (DEBUG_VERBOSE)
    {
        std::cout << "  After reorderInPlace:\n";
        for (size_t i = 0; i < data.size(); ++i)
            std::cout << "    [" << i << "] = " << data[i] << "  expected " << expected[i] << "\n";
    }

    assert(areComplexVectorsEqual(data, expected));
    std::cout << "  N=6 permutation PASSED" << std::endl;
}

void test_reorderInPlace_identityAndFixedPoints()
{
    std::cout << "Running test: test_reorderInPlace_identityAndFixedPoints..." << std::endl;

    // Identity permutation
    {
        CVector data = {Complex(1, 1), Complex(2, 2), Complex(3, 3)};
        std::vector<int> table = {0, 1, 2};
        CVector before = data;
        reorderInPlace(data, table);
        assert(areComplexVectorsEqual(data, before));
        if (DEBUG_VERBOSE)
            std::cout << "  identity PASSED\n";
    }

    // Some fixed points, some cycles
    {
        CVector data = {Complex(0, 0), Complex(1, 0), Complex(2, 0), Complex(3, 0)};
        std::vector<int> table = {0, 2, 1, 3}; // 0 and 3 fixed, 1<->2 swapped
        CVector expected(4);
        for (size_t n = 0; n < table.size(); ++n)
            expected[n] = data[table[n]];

        reorderInPlace(data, table);
        assert(areComplexVectorsEqual(data, expected));
        if (DEBUG_VERBOSE)
            std::cout << "  fixed points + swap PASSED\n";
    }

    std::cout << "  identity and fixed-points PASSED" << std::endl;
}

void test_PermutationTable()
{
    std::cout << "Running test: test_PermutationTable..." << std::endl;

    // Test N=6, {2, 3}
    std::vector<int> radices_6 = {2, 3};
    std::vector<int> table_6 = generatePermutationTable(6, radices_6);
    if (DEBUG_VERBOSE)
    {
        std::cout << "Generated Permutation Table for N=6, {2,3}:\n";
        for (size_t i = 0; i < table_6.size(); ++i)
        {
            std::cout << "  " << i << " -> " << table_6[i] << "\n";
        }
    }
    std::vector<int> expected_6 = {0, 3, 1, 4, 2, 5};
    assert(areVectorsEqual(table_6, expected_6));
    std::cout << "  N=6 {2, 3} PASSED" << std::endl;

    // Test N=8, {2, 2, 2}
    {
        std::vector<int> radices_8 = {2, 2, 2};
        std::vector<int> table_8 = generatePermutationTable(8, radices_8);
        std::vector<int> expected_8 = {0, 4, 2, 6, 1, 5, 3, 7};
        assert(areVectorsEqual(table_8, expected_8));
        if (DEBUG_VERBOSE)
            std::cout << "  N=8 {2,2,2} table PASSED\n";
    }

    // Test N=9, {3, 3}
    {
        std::vector<int> radices_9 = {3, 3};
        std::vector<int> table_9 = generatePermutationTable(9, radices_9);
        std::vector<int> expected_9 = {0, 3, 6, 1, 4, 7, 2, 5, 8};
        assert(areVectorsEqual(table_9, expected_9));
        if (DEBUG_VERBOSE)
            std::cout << "  N=9 {3,3} table PASSED\n";
    }
}

void test_invertPermutationTable()
{
    std::cout << "Running test: test_invertPermutationTable..." << std::endl;

    // Use N=6 example
    std::vector<int> radices_6 = {2, 3};
    std::vector<int> table_6 = generatePermutationTable(6, radices_6);
    std::vector<int> inv = invertPermutationTable(table_6);

    // inv[table[n]] == n
    for (size_t n = 0; n < table_6.size(); ++n)
    {
        assert(inv[table_6[n]] == static_cast<int>(n));
    }

    // table[inv[n]] == n
    for (size_t n = 0; n < table_6.size(); ++n)
    {
        assert(table_6[inv[n]] == static_cast<int>(n));
    }

    // Inverting twice should yield the original table
    std::vector<int> inv2 = invertPermutationTable(inv);
    assert(areVectorsEqual(inv2, table_6));

    if (DEBUG_VERBOSE)
        std::cout << "  invertPermutationTable checks PASSED\n";
}

int main()
{
    std::cout << "--- Running Utility Tests ---" << std::endl;

    test_PermutationTable();
    test_reorderInPlace_basicCycles();
    test_reorderInPlace_withPermutationTable();
    test_reorderInPlace_identityAndFixedPoints();
    test_invertPermutationTable();

    std::cout << "-----------------------------" << std::endl;
    std::cout << "All utility tests passed!" << std::endl;
    return 0;
}