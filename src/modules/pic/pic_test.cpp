/*
 * test_particle.cpp
 *
 *  Created on: 2012-2-9
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include <map>
#include "include/defs.h"

#include "pic/pic.h"
#include "pic/gyro_gauge.h"
#include "pic/delta_f.h"

using namespace simpla;

template<typename TPEngine>
class TestPICBasic: public testing::Test
{
protected:
	virtual void SetUp()
	{
		IVec3 dims =
		{ 20, 30, 40 };

		Vec3 xmin =
		{ 0, 0, 0 };

		Vec3 xmax =
		{ 1, 1, 1 };

		grid.initialize(1.0, xmin, xmax, dims);

		m = 2.0;
		q = 1.0;
		T = 1.0e-4;

		map["n1"] = TR1::shared_ptr<Object>(new ZeroForm(grid));
	}
public:
	static const int NDIMS = 3;
	Real m, q, T;
	DefaultGrid grid;
	typedef pic::PIC<TPEngine, DefaultGrid> ParticleType;
	std::map<std::string, TR1::shared_ptr<Object> > map;

};
typedef testing::Types<pic::DeltaF<Real>, pic::GyroGauge<Real> > AllPICEngine;

TYPED_TEST_CASE(TestPICBasic, AllPICEngine);

TYPED_TEST(TestPICBasic,create_init_load_check_T){
{
	typename TestFixture::ParticleType p(TestFixture::grid);

	p.set_property(TestFixture::m,TestFixture::q,TestFixture::T);

	p.Initialize(*TR1::dynamic_pointer_cast<ZeroForm>(TestFixture::map.find("n1")->second),100);

	Real T = 0.0;

	for(typename TestFixture::ParticleType::iterator it=p.begin();it!=p.end();++it)
	{
		for(typename TestFixture::ParticleType::Point_s *p=*it; p!=NULL; p=p->next)
		{
			T+=Dot(p->V,p->V);
		}
	}

	T*=0.5*TestFixture::m/static_cast<Real>(p.get_num_of_ele()*3);

	Real epsilon=1.0e-3;

	EXPECT_LE(fabs(T-TestFixture::T)/TestFixture::T,epsilon);

}
}

TYPED_TEST(TestPICBasic,scatter){
{

}
}
