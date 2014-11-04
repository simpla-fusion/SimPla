/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../manifold/domain_dummy.h"

#include "field.h"
#include "field_basic_algerbra_test.h"

using namespace simpla;

//int main(int argc, char **argv)
//{
//	LOGGER.set_stdout_visable_level(10);
//
//	DomainDummy<> domain(10, 20);
//
//	auto f1 = make_field<double>(domain);
//	auto f2 = make_field<double>(domain);
//	auto f3 = make_field<double>(domain);
//
//	f1 = 1.0;
//
//	f2 = 2.0;
//
//	LOGGER << "========";
//
//	auto expr = f1 + f2 * 2;
//
//	LOGGER << "========";
//
//	f3 = expr;
//
//	CHECK(f1[10]);
//	CHECK(f2[10]);
//	CHECK(f3[10]);
//
//}

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

