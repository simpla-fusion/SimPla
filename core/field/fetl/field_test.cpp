/*
 * field_test.cpp
 *
 *  created on: 2014-6-30
 *      Author: salmon
 */

//#include "field_test.h"
//
//#include "../utilities/container_dense.h"
//#include "../utilities/container_sparse.h"
//#include "../manifold/geometry/cartesian.h"
//#include "../manifold/topology/structured.h"
//#include "../manifold/manifold.h"
#include <iostream>
#include <vector>
#include <typeinfo>

#include "field.h"

using namespace simpla;

//template<typename TM> using ParamList=
//testing::Types<
//Field<Domain<TM,VERTEX> >
//>;
//
//typedef ParamList<Manifold<CartesianCoordinate<StructuredMesh>, false>,
//		DenseContainer> ParamList0d;
//typedef ParamList<Manifold<CartesianCoordinate<StructuredMesh>, false>,
//		SparseContainer> ParamList0s;
//
//typedef ParamList<Manifold<CartesianCoordinate<StructuredMesh>, true>,
//		DenseContainer> ParamList1;
//typedef ParamList<Manifold<CylindricalCoordinate<StructuredMesh>, false>,
//		DenseContainer> ParamList2;
//typedef ParamList<Manifold<CylindricalCoordinate<StructuredMesh>, true>,
//		DenseContainer> ParamList3;
//
//INSTANTIATE_TYPED_TEST_CASE_P(Cartesian_d, TestField, ParamList0d);
//INSTANTIATE_TYPED_TEST_CASE_P(Cartesian_s, TestField, ParamList0s);
//
//INSTANTIATE_TYPED_TEST_CASE_P(Cartesian_kz, TestField, ParamList1);
//INSTANTIATE_TYPED_TEST_CASE_P(Cylindrical, TestField, ParamList2);
//INSTANTIATE_TYPED_TEST_CASE_P(Cylindrical_kz, TestField, ParamList3);

struct Base
{
	typedef double type;
};
struct Derived: public Base
{

};
int main(int argc, char **argv)
{
	Field<double> a;
	std::cout << a.size() << std::endl;
	std::cout << typeid(Derived::type).name() << std::endl;
}
