/*
 * test3.cpp
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#include <iostream>
#include "fetl/ntuple.h"
#include "fetl/ntuple_ops.h"
#include "utilities/log.h"

using namespace simpla;

int main()
{
	nTuple<2, nTuple<2, double>> a =
	{ 1, 2, 3, 4 };
	nTuple<2, nTuple<2, double>> b =
	{ 0, 5, 6, 7 };
	CHECK(TensorProduct(a, b));

	nTuple<2, nTuple<2, nTuple<2, double>>> e =
	{	1,2,3,4,5,6,7,8};
	nTuple<2, nTuple<2, nTuple<2, double>>> f =
	{	1,2,3,4,5,6,7,8};
	CHECK(TensorProduct(e, f));

	nTuple<3, double> c =
	{ 1, 2, 3 };
	nTuple<1, nTuple<3, double>> d =
	{ 4, 5, 6 };
	CHECK(c);
	CHECK(d);

	CHECK(TensorProduct(c, c));
	CHECK(TensorProduct(2, c));
	CHECK(TensorProduct(c, 2));
	CHECK(TensorProduct(c, d));

	CHECK(TensorProduct(d, c));

	nTuple<3, nTuple<2, nTuple<3, double>>> g =
	{	1, 2, 3, 4, 5, 6, 7, 8, 9,1, 2, 3, 4, 5, 6, 7, 8, 9};
	CHECK(InnerProduct(g, c));

}
