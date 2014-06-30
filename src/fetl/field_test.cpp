/*
 * field_test.cpp
 *
 *  Created on: 2014年6月30日
 *      Author: salmon
 */

#include "field_test.h"

#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cartesian.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/uniform_array.h"

using namespace simpla;

template<typename TM> using ParamList=
testing::Types<
typename TM::template DenseForm<VERTEX,Real>,
typename TM::template SparseForm<VERTEX,Real>

//ParamType<TM, VERTEX, Real> //
//, ParamType<TM, EDGE, Real>//
//, ParamType<TM, FACE, Real>//
//, ParamType<TM, VOLUME, Real>//
//, ParamType<TM, VERTEX, Complex>//
//, ParamType<TM, EDGE, Complex>//
//, ParamType<TM, FACE, Complex>//
//, ParamType<TM, VOLUME, Complex>//
//, ParamType<TM, VERTEX, nTuple<3, Real> >//
//, ParamType<TM, EDGE, nTuple<3, Real> >//
//, ParamType<TM, FACE, nTuple<3, Real> >//
//, ParamType<TM, VOLUME, nTuple<3, Real> >//
//, ParamType<TM, VERTEX, nTuple<3, Complex> >//
//, ParamType<TM, EDGE, nTuple<3, Complex> >//
//, ParamType<TM, FACE, nTuple<3, Complex> >//
//, ParamType<TM, VOLUME, nTuple<3, Complex> >//

>;
typedef ParamList<Mesh<CartesianGeometry<UniformArray>, false> > ParamList0;
//typedef ParamList<Mesh<CartesianGeometry<UniformArray>, true> > ParamList1;
//typedef ParamList<Mesh<CylindricalGeometry<UniformArray>, false> > ParamList2;
//typedef ParamList<Mesh<CylindricalGeometry<UniformArray>, true> > ParamList3;

INSTANTIATE_TYPED_TEST_CASE_P(Cartesian, TestField, ParamList0);
//INSTANTIATE_TYPED_TEST_CASE_P(Cartesian_kz, TestField, ParamList1);
//INSTANTIATE_TYPED_TEST_CASE_P(Cylindrical, TestField, ParamList2);
//INSTANTIATE_TYPED_TEST_CASE_P(Cylindrical_kz, TestField, ParamList3);
