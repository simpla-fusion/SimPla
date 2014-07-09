/*
 * interpolator_test.h
 *
 *  created on: 2014-6-29
 *      Author: salmon
 */

#ifndef INTERPOLATOR_TEST_H_
#define INTERPOLATOR_TEST_H_

#include <gtest/gtest.h>

#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../physics/constants.h"
#include "../io/data_stream.h"
#include "../parallel/message_comm.h"
#include "../fetl/field.h"
#include "../utilities/container_sparse.h"

using namespace simpla;

template<typename TP>
class TestInterpolator: public testing::Test
{
protected:
	void SetUp()
	{
		LOG_STREAM.set_stdout_visable_level(10);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || xmax[i] <= xmin[i])
			{
				xmax[i] = xmin[i];
				dims[i] = 1;
			}
		}
		mesh.set_dimensions(dims);
		mesh.set_extents(xmin, xmax);
		mesh.Update();

	}
public:
	typedef typename TP::mesh_type mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::compact_index_type compact_index_type;
	typedef typename mesh_type::range_type range_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;
	static constexpr unsigned int NDIMS = mesh_type::NDIMS;
	static constexpr unsigned int IForm = TP::IForm;
	mesh_type mesh;

	coordinates_type xmin =
	{	12,0,0};

	coordinates_type xmax =
	{	14,1,1};

	nTuple<NDIMS, index_type> dims =
	{	50,30,20};

};

TYPED_TEST_CASE_P(TestInterpolator);

TYPED_TEST_P(TestInterpolator,scatter){
{

	auto const & mesh=TestFixture::mesh;
	static constexpr unsigned int IForm = TestFixture::IForm;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;
	typedef typename mesh_type::interpolator_type interpolator_type;

	auto f= mesh.template make_field<Field<mesh_type,IForm,SparseContainer<compact_index_type,scalar_type>>> ();

	typename decltype(f)::field_value_type a;

	a=1.0;

	Real w=2;

	auto extents= mesh.get_extents();

	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.1234567*(std::get<1>(extents)-std::get<0>(extents)));

	interpolator_type::Scatter(&f,std::make_tuple(x,a),w);

	scalar_type b=0;

	SparseContainer<compact_index_type,scalar_type>& g=f;

	for(auto const & v:g)
	{
		b+=v.second;
	}

	if(IForm==VERTEX || IForm==VOLUME)
	{
		EXPECT_LE(abs( w-b),EPSILON*10);
	}
	else
	{
		EXPECT_LE(abs( w*3-b),EPSILON*100);
	}

}
}

TYPED_TEST_P(TestInterpolator,gather){
{

	auto const & mesh=TestFixture::mesh;
	static constexpr unsigned int IForm = TestFixture::IForm;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;
	typedef typename mesh_type::interpolator_type interpolator_type;

	auto f= mesh.template make_field<IForm,scalar_type > ();

	f.clear();

	auto extents= mesh.get_extents();

	auto xmax= std::get<1>(extents);
	auto xmin= std::get<0>(extents);

	nTuple<NDIMS,Real> K=
	{	5/(xmax[0]-xmin[0]), 3/(xmax[1]-xmin[1]), 2/(xmax[2]-xmin[2])};

	for(auto s:mesh.Select(IForm))
	{
		f[s]= (InnerProductNTuple(K,mesh.get_coordinates(s)-xmin));
	}
	coordinates_type x = (xmin+0.34567*(xmax-xmin));

	Real expect= (InnerProductNTuple(K,x-xmin));

	Real error = abs(InnerProductNTuple(K , mesh.get_dx()));

	auto actual= interpolator_type::Gather( f ,(x));

	EXPECT_LE(abs( (actual -expect)/expect),error)<<actual <<" "<<expect;

}
}

REGISTER_TYPED_TEST_CASE_P(TestInterpolator, scatter, gather);

#endif /* INTERPOLATOR_TEST_H_ */
