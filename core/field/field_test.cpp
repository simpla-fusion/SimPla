/*
 * field_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */
#include <iostream>
#include <gtest/gtest.h>

#include "field.h"

#include "field_test1.h"

#include "../parallel/block_range.h"
using namespace simpla;

typedef testing::Types< //

		_Field<BlockRange<size_t>, std::shared_ptr<double>> //
// 	,TestFIELDParam1<VERTEX, Real>
//		, TestFIELDParam1<EDGE, Real>	//
//		, TestFIELDParam1<FACE, Real>	//
//		, TestFIELDParam1<VOLUME, Real>	//
//
//		, TestFIELDParam1<VERTEX, Complex>	//
//		, TestFIELDParam1<EDGE, Complex>	//
//		, TestFIELDParam1<FACE, Complex>	//
//		, TestFIELDParam1<VOLUME, Complex>	//
//
//		, TestFIELDParam1<VERTEX, nTuple<3, Real> >	//
//		, TestFIELDParam1<EDGE, nTuple<3, Real> >	//
//		, TestFIELDParam1<FACE, nTuple<3, Real> >	//
//		, TestFIELDParam1<VOLUME, nTuple<3, Real> >	//
//
//		, TestFIELDParam1<VERTEX, nTuple<3, Complex> >	//
//		, TestFIELDParam1<EDGE, nTuple<3, Complex> >	//
//		, TestFIELDParam1<FACE, nTuple<3, Complex> >	//
//		, TestFIELDParam1<VOLUME, nTuple<3, Complex> >	//
//
//		, TestFIELDParam1<VERTEX, nTuple<3, nTuple<3, Real>> >	//
//		, TestFIELDParam1<EDGE, nTuple<3, nTuple<3, Real>> >	//
//		, TestFIELDParam1<FACE, nTuple<3, nTuple<3, Real>> >	//
//		, TestFIELDParam1<VOLUME, nTuple<3, nTuple<3, Real>> >	//
//
//		, TestFIELDParam1<VERTEX, nTuple<3, nTuple<3, Complex>> >	//
//		, TestFIELDParam1<EDGE, nTuple<3, nTuple<3, Complex>> >	//
//		, TestFIELDParam1<FACE, nTuple<3, nTuple<3, Complex>> >	//
//		, TestFIELDParam1<VOLUME, nTuple<3, nTuple<3, Complex>> >	//

> TypeParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

