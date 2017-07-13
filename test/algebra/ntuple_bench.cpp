//
// Created by salmon on 17-1-1.
//
#include <benchmark/benchmark.h"
#include "simpla/algebra/nTuple.h"
#include <eigen3/Eigen/Core>

using namespace simpla;
typedef double Real;

static constexpr int a = 1, b = 2, c = 3;

#define EQUATION(_A, _B, _C) (-(_A * a) + (_B * b / c) - _C)

template <typename value_type, int N>
static void BM_raw_array(benchmark::State &state) {
    value_type aA[N], aB[N], aC[N], aD[N], res[N];
    for (int i = 0; i < N; ++i) {
        aA[i] = i;
        aB[i] = i;
        aC[i] = i;
        aD[i] = i;
    }

    while (state.KeepRunning()) {
        for (int i = 0; i < N; ++i) { aD[i] += EQUATION(aA[i], aB[i], aC[i]); }
    }

    static value_type tmp;
    benchmark::DoNotOptimize(tmp = aD[N - 1]);
}

template <typename value_type, int... N>
static void BM_nTuple(benchmark::State &state) {
    nTuple<value_type, N...> vA, vB, vC, vD;

    vA = 1;
    vB = 1;
    vC = 1;
    vD = 1;

    while (state.KeepRunning()) { vD += EQUATION(vA, vB, vC); }

    static value_type tmp;
    static constexpr int s = 0;
    benchmark::DoNotOptimize(tmp = vD.at(&s));
}

template <typename value_type, int... N>
static void BM_eigen(benchmark::State &state) {
    Eigen::Matrix<value_type, N...> vA, vB, vC, vD;

    while (state.KeepRunning()) { vD += EQUATION(vA, vB, vC); }
}

BENCHMARK_TEMPLATE(BM_raw_array, double, 3);
BENCHMARK_TEMPLATE(BM_nTuple, double, 3);
BENCHMARK_TEMPLATE(BM_eigen, double, 3, 1);
BENCHMARK_TEMPLATE(BM_eigen, double, 1, 3);
BENCHMARK_TEMPLATE(BM_raw_array, double, 9);
BENCHMARK_TEMPLATE(BM_nTuple, double, 9);
BENCHMARK_TEMPLATE(BM_eigen, double, 9, 1);
BENCHMARK_TEMPLATE(BM_eigen, double, 1, 9);
BENCHMARK_TEMPLATE(BM_nTuple, double, 3, 3);
BENCHMARK_TEMPLATE(BM_eigen, double, 3, 3);

BENCHMARK_TEMPLATE(BM_raw_array, double, 27);
BENCHMARK_TEMPLATE(BM_nTuple, double, 27);
BENCHMARK_TEMPLATE(BM_eigen, double, 27, 1);
BENCHMARK_TEMPLATE(BM_eigen, double, 1, 27);
BENCHMARK_TEMPLATE(BM_nTuple, double, 3, 3, 3);

BENCHMARK_TEMPLATE(BM_raw_array, double, 100);
BENCHMARK_TEMPLATE(BM_nTuple, double, 100);
BENCHMARK_TEMPLATE(BM_eigen, double, 100, 1);
BENCHMARK_TEMPLATE(BM_eigen, double, 1, 100);

BENCHMARK_TEMPLATE(BM_nTuple, double, 10, 10);
BENCHMARK_TEMPLATE(BM_eigen, double, 10, 10);

BENCHMARK_TEMPLATE(BM_raw_array, int, 3);
BENCHMARK_TEMPLATE(BM_nTuple, int, 3);
BENCHMARK_TEMPLATE(BM_eigen, int, 3, 1);
BENCHMARK_TEMPLATE(BM_eigen, int, 1, 3);
BENCHMARK_TEMPLATE(BM_raw_array, int, 9);
BENCHMARK_TEMPLATE(BM_nTuple, int, 9);
BENCHMARK_TEMPLATE(BM_eigen, int, 9, 1);
BENCHMARK_TEMPLATE(BM_eigen, int, 1, 9);
BENCHMARK_TEMPLATE(BM_nTuple, int, 3, 3);
BENCHMARK_TEMPLATE(BM_eigen, int, 3, 3);
BENCHMARK_TEMPLATE(BM_raw_array, int, 27);
BENCHMARK_TEMPLATE(BM_nTuple, int, 27);
BENCHMARK_TEMPLATE(BM_eigen, int, 27, 1);
BENCHMARK_TEMPLATE(BM_eigen, int, 1, 27);
BENCHMARK_TEMPLATE(BM_nTuple, int, 3, 3, 3);
BENCHMARK_TEMPLATE(BM_raw_array, int, 100);
BENCHMARK_TEMPLATE(BM_nTuple, int, 100);
BENCHMARK_TEMPLATE(BM_eigen, int, 100, 1);
BENCHMARK_TEMPLATE(BM_eigen, int, 1, 100);
BENCHMARK_TEMPLATE(BM_nTuple, int, 10, 10);
BENCHMARK_TEMPLATE(BM_eigen, int, 10, 10);

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}