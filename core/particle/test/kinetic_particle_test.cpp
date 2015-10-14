/**
 * @file kinetic_particle_test.cpp
 *
 * @date 2015-2-11
 * @author salmon
 */

#include <gtest/gtest.h>
#include "../kinetic_particle.h"
#include "../simple_particle.h"
#include "rect_mesh.h"
#include "../../field/field.h"

#include "../../utilities/memory_pool.h"
#include "../../gtl/iterator/sp_iterator.h"
#include "../../io/data_stream.h"
#include "../../io/io.h"

using namespace simpla;

class TestKineticParticle: public testing::Test
{
protected:
	virtual void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_DEBUG);
		nTuple<Real, 3> xmin =
		{ 0, 2, 3 };
		nTuple<Real, 3> xmax =
		{ 11, 15, 16 };
		nTuple<size_t, 3> imin =
		{ 0, 0, 0 };
		nTuple<size_t, 3> imax =
		{ 10, 10, 10 };
		mesh = std::shared_ptr<mesh_type>(
				new mesh_type(xmin, xmax, imin, imax));
	}
public:
	typedef SimpleMesh mesh_type;
	typedef SimpleParticleEngine engine_type;
	typedef typename mesh_type::coordinate_tuple coordinate_tuple;

	typedef KineticParticle<mesh_type, engine_type> particle_type;

	static constexpr size_t pic = 10;

	std::shared_ptr<mesh_type> mesh;

	typedef typename engine_type::Point_s Point_s;

};
constexpr size_t TestKineticParticle::pic;

TEST_F(TestKineticParticle,insert)
{

	particle_type p(*mesh);

	auto extents = mesh->extents();

	auto range = mesh->range();

	auto p_generator = simple_particle_generator(p, extents, 1.0);

	std::mt19937 rnd_gen;

	size_t num = 1000; //range.size();

	std::copy(p_generator.begin(rnd_gen), p_generator.end(rnd_gen, pic * num),
			std::front_inserter(p));

	Real variance = 0;
	Real mean = 0;

	for (typename particle_type::bucket_type const & item : p.select(range))
	{
		size_t n = std::distance(item.begin(), item.end());
		variance += (static_cast<Real>(n) - pic) * (static_cast<Real>(n) - pic);
	}

	variance = std::sqrt(variance / (num * pic * pic));

	EXPECT_LE(variance, 1.0 / std::sqrt(pic));

	CHECK(variance);

	CHECK(p.size());

	EXPECT_EQ(p.size(), pic * num);

	LOGGER << save("pic", p.dataset()) << std::endl;

}

//TEST_F(TestKineticParticle, dump)
//{
//
//	SimpleField<mesh_type, Real> n(manifold), n0(manifold);
//
//	particle_type ion(manifold);
//
//	SimpleField<mesh_type, Real> E(manifold);
//	SimpleField<mesh_type, Real> B(manifold);
//
//	E.clear();
//	B.clear();
//	n0.clear();
//	n.clear();
//
//	Real q = ion.charge;
//	Real variance = 0.0;
//
//	Real average = 0.0;
//
////	for (auto s : manifold->range())
////	{
////		coordinate_tuple x = manifold->id_to_coordinates(s);
////
////		Real expect = q * n(x[0], x[1], x[2]).template as<Real>();
////
////		n0[s] = expect;
////
////		Real actual = n.get(s);
////
////		average += abs(actual);
////
////		variance += std::pow(abs(expect - actual), 2.0);
////	}
//
////	if (std::is_same<engine_type, PICEngineFullF<mesh_type> >::value)
////	{
////		Real relative_error = std::sqrt(variance) / abs(average);
////		CHECK(relative_error);
////		EXPECT_LE(relative_error, 1.0 / std::sqrt(pic));
////	}
////	else
//	{
//		Real error = 1.0 / std::sqrt(static_cast<double>(ion.size()));
//
//		EXPECT_LE(abs(average), error);
//	}
//
//}
//
