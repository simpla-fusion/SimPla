//
// Created by salmon on 17-1-1.
//
#include <simpla/algebra/nTuple.h>
#include <benchmark/benchmark.h>

using namespace simpla;

//static constexpr int N = 3;
//typedef double value_type;
static constexpr Real a = 1, b = 2, c = 3;

#define EQUATION(_A, _B, _C)  ( -(_A  +a )/(   _B *b -c  )- _C)

//#define BENCHMARK_TEMPLATE(_NAME_, _T_, _N_) \
//static void _NAME_##_T_##_N_(benchmark::State &state) { _NAME_<_T_,_N_>(state);} \
//BENCHMARK(_NAME_##_T_##_N_);

// Define another benchmark
template<typename value_type, size_type N>
static void BM_raw_array(benchmark::State &state)
{
    value_type aA[N], aB[N], aC[N], aD[N], res[N];
    for (int i = 0; i < N; ++i)
    {
        aA[i] = i;
        aB[i] = i;
        aC[i] = i;
        aD[i] = i;
    }

    while (state.KeepRunning()) { for (int i = 0; i < N; ++i) { aD[i] += EQUATION(aA[i], aB[i], aC[i]); }}

    static value_type tmp;
    benchmark::DoNotOptimize(tmp = aD[N - 1]);
}


// Define another benchmark
template<typename value_type, size_type ...N>
static void BM_nTuple(benchmark::State &state)
{
    nTuple<value_type, N...> vA, vB, vC, vD;

    vA = 1;
    vB = 1;
    vC = 1;
    vD = 1;

    while (state.KeepRunning()) { vD += EQUATION(vA, vB, vC); }

    static value_type tmp;
    static constexpr size_type s = 0;
    benchmark::DoNotOptimize(tmp = vD.at(&s));

}


BENCHMARK_TEMPLATE(BM_raw_array, double, 3);
BENCHMARK_TEMPLATE(BM_nTuple, double, 3);


BENCHMARK_TEMPLATE(BM_raw_array, double, 10);
BENCHMARK_TEMPLATE(BM_nTuple, double, 10);

BENCHMARK_TEMPLATE(BM_raw_array, double, 100);
BENCHMARK_TEMPLATE(BM_nTuple, double, 100);

BENCHMARK_TEMPLATE(BM_raw_array, int, 3);
BENCHMARK_TEMPLATE(BM_nTuple, int, 3);

BENCHMARK_TEMPLATE(BM_raw_array, int, 10);
BENCHMARK_TEMPLATE(BM_nTuple, int, 10);

BENCHMARK_TEMPLATE(BM_raw_array, int, 100);
BENCHMARK_TEMPLATE(BM_nTuple, int, 100);


BENCHMARK_TEMPLATE(BM_nTuple, double, 3, 3);
BENCHMARK_TEMPLATE(BM_nTuple, int, 10, 10);
BENCHMARK_TEMPLATE(BM_nTuple, double, 3, 3, 3);

int main(int argc, char **argv)
{
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}