/*
 * sparse_vector_test.cpp
 *
 *  Created on: 2013年7月28日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include <complex>
#include <map>
#include "equation.h"
#include "expression.h"
#include "mesh/uniform_rect.h"
#include "fetl.h"

using namespace simpla;

DEFINE_FIELDS(UniformRectMesh)

template<typename TF>
class TestFETLBasicArithmetic: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

		mesh.dt = 1.0;
		mesh.xmin[0] = 0;
		mesh.xmin[1] = 0;
		mesh.xmin[2] = 0;
		mesh.xmax[0] = 1.0;
		mesh.xmax[1] = 1.0;
		mesh.xmax[2] = 1.0;
		mesh.dims[0] = 20;
		mesh.dims[1] = 30;
		mesh.dims[2] = 40;
		mesh.gw[0] = 2;
		mesh.gw[1] = 2;
		mesh.gw[2] = 2;

		mesh.Update();

	}
public:
	typedef TF FieldType;

	Mesh mesh;

};

typedef testing::Types<
//		RZeroForm, ROneForm, RTwoForm, RThreeForm, CZeroForm,
//		COneForm, CTwoForm, CThreeForm, VecZeroForm,
		VecOneForm
//, VecTwoForm
//		,VecThreeForm
//		,CVecZeroForm, CVecOneForm, CVecTwoForm,CVecThreeForm
> AllFieldTypes;

//, VecThreeForm

TYPED_TEST_CASE(TestFETLBasicArithmetic, AllFieldTypes);

TYPED_TEST(TestFETLBasicArithmetic,assign){
{

	std::map<size_t, double> m;

	PlaceHolder x1(14);
	PlaceHolder x2(15);
	double b=0;
	((-(x1 + x2+2 ) / 2.0 + 5 + 4.25 * (-x1 * 3 - x2 * 2) * 2+5 ) ).assign(m, b,1);

	for (int i = 0; i < 10; ++i)
	{
		PlaceHolder(i).assign(m,b);
	}
	for (auto it = m.begin(); it != m.end(); ++it)
	{
		std::cout << "[" << (it->first) << "," << (it->second) << "]"
		<< std::endl;
	}
	std::cout<<b<<std::endl;
}
}
TYPED_TEST(TestFETLBasicArithmetic,field){
{

	std::map<size_t, double> m;

	typename ColneField<typename TestFixture::FieldType,PlaceHolderGenerator>::type x;

	double b=0;
	((-(x1 + x2+2 ) / 2.0 + 5 + 4.25 * (-x1 * 3 - x2 * 2) * 2+5 ) ).assign(m, b,1);

	for (int i = 0; i < 10; ++i)
	{
		PlaceHolder(i).assign(m,b);
	}
	for (auto it = m.begin(); it != m.end(); ++it)
	{
		std::cout << "[" << (it->first) << "," << (it->second) << "]"
		<< std::endl;
	}
	std::cout<<b<<std::endl;
}
}
