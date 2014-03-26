/*
 * fetl_test4.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST4_H_
#define FETL_TEST4_H_

#include "fetl_test.h"

#include <gtest/gtest.h>
#include <random>

#include "../io/data_stream.h"
#include "save_field.h"

#include "fetl.h"
#include "ntuple.h"
#include "../mesh/traversal.h"

using namespace simpla;

template<typename TParam>
class TestFETLVecField: public testing::Test
{
protected:
	virtual void SetUp()
	{
		TParam::SetUpMesh(&mesh);
		TParam::SetDefaultValue(&default_value);
	}
public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename TParam::value_type value_type;
	typedef nTuple<3, value_type> Vec3;
	typedef Field<mesh_type, VERTEX, value_type> ScalarField;
	typedef Field<mesh_type, VERTEX, nTuple<3, value_type> > VectorField;

	mesh_type mesh;
	value_type default_value;
};

TYPED_TEST_CASE_P(TestFETLVecField);

TYPED_TEST_P(TestFETLVecField,vec_zero_form){
{
	typename TestFixture::mesh_type const & mesh = TestFixture::mesh;

	typename TestFixture::Vec3 vc1 =
	{	1.0, 2.0, 3.0};

	typename TestFixture::Vec3 vc2 =
	{	-1.0, 4.0, 2.0};

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	typename TestFixture::ScalarField res_scalar_field(mesh);

	typename TestFixture::VectorField vaf(mesh), vbf(mesh),

	res_vector_field (mesh);

	vaf.Init();
	vbf.Init();

	for(auto & p:vaf)
	{
		p = vc1*uniform_dist(gen);
	}
	for(auto & p:vbf)
	{
		p =vc2*uniform_dist(gen);
	}

	LOG_CMD(res_vector_field = Cross( vaf,vbf) );

	for(auto s:mesh.GetRegion(VERTEX))
	{
		ASSERT_EQ(Cross(vaf[s],vbf[s]), res_vector_field [s]);

	}

	LOG_CMD(res_scalar_field = Dot(vaf, vbf));

	for(auto s:mesh.GetRegion(VERTEX))
	{
		ASSERT_EQ(Dot(vaf[s],vbf[s]),res_scalar_field[s]);
	}

}
}
REGISTER_TYPED_TEST_CASE_P(TestFETLVecField, vec_zero_form);
#endif /* FETL_TEST4_H_ */
