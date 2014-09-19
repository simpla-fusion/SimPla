/*
 * field_test.cpp
 *
 *  created on: 2014-6-30
 *      Author: salmon
 */

#include "field_test.h"

#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cartesian.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/uniform_array.h"
#include "../utilities/container_dense.h"
#include "../utilities/container_sparse.h"

using namespace simpla;

template<typename TM, template<typename, typename > class Container> using ParamList=
testing::Types<
Field<TM,VERTEX,Container<typename TM::compact_index_type,Real> >

,Field<TM,EDGE,Container<typename TM::compact_index_type,Real> >

,Field<TM,FACE,Container<typename TM::compact_index_type,Real> >

,Field<TM,VOLUME,Container<typename TM::compact_index_type,Real> >

,Field<TM,VERTEX,Container<typename TM::compact_index_type,Complex> >

,Field<TM,EDGE,Container<typename TM::compact_index_type,Complex> >

,Field<TM,FACE,Container<typename TM::compact_index_type,Complex> >

,Field<TM,VOLUME,Container<typename TM::compact_index_type,Complex> >

,Field<TM,VERTEX,Container<typename TM::compact_index_type,nTuple<3,Real>> >

,Field<TM,EDGE,Container<typename TM::compact_index_type,nTuple<3,Real>> >

,Field<TM,FACE,Container<typename TM::compact_index_type,nTuple<3,Real>> >

,Field<TM,VOLUME,Container<typename TM::compact_index_type,nTuple<3,Real>> >

,Field<TM,VERTEX,Container<typename TM::compact_index_type,nTuple<3,Complex>> >

,Field<TM,EDGE,Container<typename TM::compact_index_type,nTuple<3,Complex>> >

,Field<TM,FACE,Container<typename TM::compact_index_type,nTuple<3,Complex>> >

,Field<TM,VOLUME,Container<typename TM::compact_index_type,nTuple<3,Complex> > >

>;

typedef ParamList<Mesh<CartesianCoordinates<SurturedMesh>, false>, DenseContainer> ParamList0d;
typedef ParamList<Mesh<CartesianCoordinates<SurturedMesh>, false>, SparseContainer> ParamList0s;

typedef ParamList<Mesh<CartesianCoordinates<SurturedMesh>, true>, DenseContainer> ParamList1;
typedef ParamList<Mesh<CylindricalCoordinates<SurturedMesh>, false>, DenseContainer> ParamList2;
typedef ParamList<Mesh<CylindricalCoordinates<SurturedMesh>, true>, DenseContainer> ParamList3;

INSTANTIATE_TYPED_TEST_CASE_P(Cartesian_d, TestField, ParamList0d);
INSTANTIATE_TYPED_TEST_CASE_P(Cartesian_s, TestField, ParamList0s);

INSTANTIATE_TYPED_TEST_CASE_P(Cartesian_kz, TestField, ParamList1);
INSTANTIATE_TYPED_TEST_CASE_P(Cylindrical, TestField, ParamList2);
INSTANTIATE_TYPED_TEST_CASE_P(Cylindrical_kz, TestField, ParamList3);
