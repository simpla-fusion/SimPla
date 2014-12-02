/*
 * interpolator_test.h
 *
 *  created on: 2014-6-29
 *      Author: salmon
 */

#ifndef INTERPOLATOR_TEST_H_
#define INTERPOLATOR_TEST_H_
#include <gtest/gtest.h>

#include "interpolator.h"

#include "../../data_structure/container_sparse.h"
#include "../../utilities/utilities.h"
#include "../../field/field.h"

using namespace simpla;

template<typename TF>
class TestInterpolator: public testing::Test
{
protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(10);
		xmin = coordinates_type( { 12, 0, 0 });
		xmax = coordinates_type( { 14, 1, 1 });

		dims = nTuple<index_type, NDIMS>( { 50, 30, 20 });

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || xmax[i] <= xmin[i])
			{
				xmax[i] = xmin[i];
				dims[i] = 1;
			}
		}
		manifold.dimensions(dims);
		manifold.extents(xmin, xmax);
		manifold.update();

	}
public:
	typedef typename TF::interpolator_type interpolator_type;
	typedef typename interpolator_type::manifold_type mainfold_type;
	typedef typename mainfold_type::index_type index_type;
	typedef typename mainfold_type::compact_index_type compact_index_type;
	typedef typename mainfold_type::range_type range_type;
	typedef typename mainfold_type::iterator iterator;
	typedef typename mainfold_type::coordinates_type coordinates_type;
	typedef typename mainfold_type::scalar_type scalar_type;

	static constexpr unsigned int NDIMS = mainfold_type::NDIMS;

	static constexpr unsigned int iform = TF::iform;

	mainfold_type manifold;

	coordinates_type xmin/* = { 12, 0, 0 }*/;

	coordinates_type xmax/* = { 14, 1, 1 }*/;

	nTuple<index_type, NDIMS> dims/* = { 50, 30, 20 }*/;

};

TYPED_TEST_CASE_P(TestInterpolator);

TYPED_TEST_P(TestInterpolator,scatter){
{

	auto const & mesh=TestFixture::mesh;
	static constexpr unsigned int iform = TestFixture::iform;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;
	typedef typename TestFixture::interpolator_type interpolator_type;

	auto f= mesh.template make_field<Field<mesh_type,iform,
	SparseContainer<compact_index_type,scalar_type>>> ();

	typename decltype(f)::field_value_type a;

	a=1.0;

	Real w=2;

	auto extents= mesh.get_extents();

	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+
			0.1234567*(std::get<1>(extents)-std::get<0>(extents)));

	interpolator_type::scatter(&f,std::make_tuple(x,a),w);

	scalar_type b=0;

	SparseContainer<compact_index_type,scalar_type>& g=f;

	for(auto const & v:g)
	{
		b+=v.second;
	}

	if(iform==VERTEX || iform==VOLUME)
	{
		EXPECT_LE(abs( w-b),EPSILON*10);
	}
	else
	{
		EXPECT_LE(abs( w*3-b),EPSILON*10);
	}

}
}

TYPED_TEST_P(TestInterpolator,gather){
{

	auto const & mesh=TestFixture::mesh;
	static constexpr unsigned int iform = TestFixture::iform;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;
	typedef typename TestFixture::interpolator_type interpolator_type;

	auto f= mesh.template make_field<iform,scalar_type > ();

	f.clear();

	auto extents= mesh.get_extents();

	auto xmax= std::get<1>(extents);
	auto xmin= std::get<0>(extents);

	nTuple<Real,NDIMS> K=
	{	5/(xmax[0]-xmin[0]), 3/(xmax[1]-xmin[1]), 2/(xmax[2]-xmin[2])};

	for(auto s:mesh.select(iform))
	{
		f[s]= (dot(K,mesh.get_coordinates(s)-xmin));
	}
	coordinates_type x = (xmin+0.34567*(xmax-xmin));

	Real expect= (dot(K,x-xmin));

	Real error = abs(dot(K , mesh.get_dx()));

	auto actual= interpolator_type::Gather( f ,(x));

	EXPECT_LE(abs( (actual -expect)/expect),error)<<actual <<" "<<expect;

}
}

REGISTER_TYPED_TEST_CASE_P(TestInterpolator, scatter, gather);

#endif /* INTERPOLATOR_TEST_H_ */
