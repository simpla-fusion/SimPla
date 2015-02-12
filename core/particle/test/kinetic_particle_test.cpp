/**
 * @file kinetic_particle_test.cpp
 *
 * @date 2015年2月11日
 * @author salmon
 */

#include <gtest/gtest.h>
#include "../kinetic_particle.h"
#include "../particle_engine.h"
#include "../particle_generator.h"
#include "../../mesh/simple_mesh.h"
#include "../../field/field.h"

#include "../../numeric/rectangle_distribution.h"
#include "../../numeric/multi_normal_distribution.h"
#include "../../utilities/memory_pool.h"
#include "../../gtl/iterator/sp_iterator.h"

using namespace simpla;

class TestKineticParticle: public testing::Test
{
protected:
	virtual void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_DEBUG);
	}
public:
	typedef SimpleMesh mesh_type;
	typedef SimpleParticleEngine engine_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef KineticParticle<mesh_type, engine_type> particle_type;

	static constexpr size_t pic = 10;

	mesh_type mesh;

	typedef typename engine_type::Point_s Point_s;

};
constexpr size_t TestKineticParticle::pic;

TEST_F(TestKineticParticle,insert)
{

	particle_type p(mesh);

	auto extents = mesh.extents();

	auto range = mesh.range();

	auto p_generator = make_particle_generator(p,

	rectangle_distribution<mesh_type::ndims>(extents),

	multi_normal_distribution<mesh_type::ndims>()

	);

	std::mt19937 rnd_gen;

	size_t num = (mesh.hash(*end(range)) - mesh.hash(*begin(range)));

	std::copy(p_generator.begin(rnd_gen), p_generator.end(rnd_gen, pic * num),
			std::front_inserter(p));

	Real variance = 0;
	Real mean = 0;

	for (auto s : range)
	{
		size_t n = p.size(mesh.hash(s));

		variance += (static_cast<Real>(n) - pic) * (static_cast<Real>(n) - pic);
	}

	variance = std::sqrt(variance / (num * pic * pic));

	EXPECT_LE(variance, 1.0 / std::sqrt(pic));

	CHECK(variance);

	EXPECT_EQ(p.size(), pic * num);

	EXPECT_LE(p.dataset().dataspace.size(), pic * num);

}

TEST_F(TestKineticParticle, dump)
{

	SimpleField<mesh_type, Real> n(mesh), n0(mesh);

	particle_type ion(mesh);

	SimpleField<mesh_type, Real> E(mesh);
	SimpleField<mesh_type, Real> B(mesh);

	E.clear();
	B.clear();
	n0.clear();
	n.clear();

	Real q = ion.charge;
	Real variance = 0.0;

	Real average = 0.0;

//	for (auto s : mesh.range())
//	{
//		coordinates_type x = mesh.id_to_coordinates(s);
//
//		Real expect = q * n(x[0], x[1], x[2]).template as<Real>();
//
//		n0[s] = expect;
//
//		Real actual = n.get(s);
//
//		average += abs(actual);
//
//		variance += std::pow(abs(expect - actual), 2.0);
//	}

//	if (std::is_same<engine_type, PICEngineFullF<mesh_type> >::value)
//	{
//		Real relative_error = std::sqrt(variance) / abs(average);
//		CHECK(relative_error);
//		EXPECT_LE(relative_error, 1.0 / std::sqrt(pic));
//	}
//	else
	{
		Real error = 1.0 / std::sqrt(static_cast<double>(ion.size()));

		EXPECT_LE(abs(average), error);
	}

}

