/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include "field_basic_algerbra_test.h"

#include <iostream>
#include <gtest/gtest.h>

#include "field.h"
#include "domain_dummy.h"

using namespace simpla;

typedef testing::Types< //

		Field<double, DomainDummy<>> //
//		, Field<nTuple<double, 3>, DomainDummy<>> //

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

