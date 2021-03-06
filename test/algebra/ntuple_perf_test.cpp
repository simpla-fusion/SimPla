/**
 * @file ntuple_perf_test.cpp
 *
 *  Created on: 2015-6-21
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include "simpla/algebra/nTuple.h"

using namespace simpla;

#define EQUATION(_A, _B, _C)  ( -(_A  +TestFixture::a )/(   _B *TestFixture::b -TestFixture::c  )- _C)

template<typename> class nTuplePerf1;

template<typename T, size_t N>
class nTuplePerf1<nTuple<T, N>> : public testing::Test
{
    virtual void SetUp()
    {

        a = 1;
        b = 3;
        c = 4;
        d = 7;

        dims0 = N;

        for (int i = 0; i < dims0; ++i)
        {
            aA[i] = i * 2;
            aB[i] = 5 - i;
            aC[i] = i * 5 + 1;
            aD[i] = 0;
            vA[i] = aA[i];
            vB[i] = aB[i];
            vC[i] = aC[i];
            vD[i] = 0;

            res[i] = -(aA[i] + a) / (aB[i] * b - c) - aC[i];
        }

    }

public:
    std::size_t num_of_loops = 10000000L;

    typedef nTuple<T, N> type;

    typedef T value_type;

    type vA, vB, vC, vD;

    value_type aA[N], aB[N], aC[N], aD[N], res[N];

    value_type a, b, c, d;

    size_t dims0, dims1;

};

typedef testing::Types<

        nTuple<double, 3>, nTuple<int, 3>, nTuple<double, 20>, nTuple<int, 10>, nTuple<std::complex<double>, 3>, nTuple<std::complex<double>, 10>

> ntuple_type_lists_1d;

TYPED_TEST_CASE(nTuplePerf1, ntuple_type_lists_1d);

TYPED_TEST(nTuplePerf1, performance_raw_array)
{

    for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
    {
        for (int i = 0; i < TestFixture::dims0; ++i)
        {
            TestFixture::aD[i] += EQUATION(TestFixture::aA[i], TestFixture::aB[i], TestFixture::aC[i]);
            //*                                 static_cast<typename TestFixture::value_type>(s);

        }

    }

}

TYPED_TEST(nTuplePerf1, performancenTuple)
{


    for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
    {
        TestFixture::vD += EQUATION(TestFixture::vA, TestFixture::vB, TestFixture::vC);
        //*                        static_cast<typename TestFixture::value_type>(s);

    }


}

template<typename> class nTuplePerf2;

template<typename T, size_t N, size_t M>
class nTuplePerf2<nTuple<T, M, N>> : public testing::Test
{
    virtual void SetUp()
    {

        a = 1;
        b = 3;
        c = 4;
        d = 7;

        dims0 = N;
        dims1 = M;

        for (int i = 0; i < dims0; ++i)
            for (int j = 0; j < dims1; ++j)
            {
                aA[i][j] = i * 2;
                aB[i][j] = 5 - i;
                aC[i][j] = i * 5 + 1;
                aD[i][j] = 0;
                vA[i][j] = aA[i][j];
                vB[i][j] = aB[i][j];
                vC[i][j] = aC[i][j];
                vD[i][j] = 0;

                res[i][j] = -(aA[i][j] + a) / (aB[i][j] * b - c) - aC[i][j];
            }

    }

public:
    std::size_t num_of_loops = 10000000L;

    typedef Tensor<T, N, M> type;

    typedef T value_type;

    type vA, vB, vC, vD;

    value_type aA[N][M], aB[N][M], aC[N][M], aD[N][M], res[N][M];

    value_type a, b, c, d;

    size_t dims0, dims1;

};

typedef testing::Types<
        Tensor<double, 3, 4>             //
        , Tensor<double, 20, 10>                 //
        , Tensor<int, 3, 4>                      //
        , Tensor<int, 10, 20>                    //
        , Tensor<std::complex<double>, 3, 4>     //
        , Tensor<std::complex<double>, 10, 20>   //

> ntuple_type_lists_2d;

TYPED_TEST_CASE(nTuplePerf2, ntuple_type_lists_2d);

TYPED_TEST(nTuplePerf2, performance_raw_array)
{

    for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
    {
        for (int i = 0; i < TestFixture::dims0; ++i)
            for (int j = 0; j < TestFixture::dims1; ++j)
            {
                TestFixture::aD[i][j] +=
                        EQUATION(TestFixture::aA[i][j], TestFixture::aB[i][j], TestFixture::aC[i][j]) *
                        static_cast<typename TestFixture::value_type>(s);
            }

    }

}

TYPED_TEST(nTuplePerf2, performancenTuple)
{
    for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
    {
        TestFixture::vD += EQUATION(TestFixture::vA, TestFixture::vB, TestFixture::vC) *
                           static_cast<typename TestFixture::value_type>(s);

    }
}
