/**
 * @file kinetic_particle_test.cpp
 *
 * @date 2015年2月11日
 * @author salmon
 */

#include <gtest/gtest.h>
#include "../kinetic_particle.h"
#include "../particle_engine.h"
#include "../../mesh/simple_mesh.h"
#include "../../field/field.h"
#include "../../numeric/rectangle_distribution.h"

using namespace simpla;

class TestKineticParticle: public testing::TestWithParam<SimpleMesh>
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
	mesh_type m_mesh;

	typedef KineticParticle<mesh_type, engine_type> particle_type;

	mesh_type mesh;

	nTuple<Real, 3> xmin, xmax;

	nTuple<size_t, 3> dims;

};

TEST_P(TestKineticParticle,Add)
{

	particle_type p(mesh);

	auto extents = mesh.extents();

	rectangle_distribution<mesh_type::ndims> x_dist(extents.first,
			extents.second);

	std::mt19937 rnd_gen(mesh_type::ndims);

	nTuple<Real, 3> v = { 0, 0, 0 };

	int pic = 100;

	for (auto s : mesh.range())
	{
		for (int i = 0; i < pic; ++i)
		{
			p.push_back(s, SimpleParticleEngine::Point_s( { x_dist(rnd_gen), v,
					1.0 }));
		}
	}

//	INFORM << "Add particle DONE " << p.size() << std::endl;
//
//	EXPECT_EQ(p.size(), mesh.get_local_memory_size(VERTEX) * pic);

//	sync(&p);

//	VERBOSE << "update_ghosts particle DONE " << p.size() << std::endl;

}

TEST_P(TestKineticParticle, scatter_n)
{

	SimpleField<mesh_type, Real> n(mesh), n0(mesh);

	particle_type ion(mesh);

	SimpleField<mesh_type, Real> E(mesh);
	SimpleField<mesh_type, Real> B(mesh);

	E.clear();
	B.clear();
	n0.clear();
	n.clear();

//	scatter(ion, &n, E, B);

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
