/**
 * \file particle_source.h
 *
 * \date    2014年9月2日  上午10:00:41 
 * \author salmon
 */

#ifndef PARTICLE_SOURCE_H_
#define PARTICLE_SOURCE_H_

#include <tuple>

#include "../numeric/multi_normal_distribution.h"
#include "../numeric/rectangle_distribution.h"
#include "../physics/physical_constants.h"
#include "../utilities/log.h"
#include "../parallel/mpi_aux_functions.h"
#include "../utilities/misc_utilities.h"

namespace simpla
{

template<typename TM, typename Engine, typename Policy = std::nullptr_t>
class ParticleSource
{
public:
	typedef TM mesh_type;
	typedef Engine engine_type;
	typedef typename engine_type::Poiont_s value_type;
	typedef ParticleSource<mesh_type, engine_type> this_type;
	mesh_type const & mesh;
	engine_type const & engine;
private:
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::compact_index_type compact_index_type;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	std::mt19937 rnd_gen;
	rectangle_distribution<NDIMS> x_dist;
	multi_normal_distribution<NDIMS> v_dist;

	Real inv_sample_density_;
	compact_index_type s_;

	Real mass_;
	Real charge_;

	std::function<Real(coordinates_type const &)> f_;
	std::function<Real(coordinates_type const &)> vs_;
public:
	template<typename ...Others>
	ParticleSource(mesh_type const &pm, engine_type pe, Others && ... others)
			: mesh(pm), engine(pe), rnd_gen(NDIMS * 2), inv_sample_density_(1.0), s_(0)
	{
		auto prop = pe.get_properties();
		mass_ = std::get<0>(prop);
		charge_ = std::get<1>(prop);
		set_distribution(std::forward<Others>(others)...);
	}
	~ParticleSource()
	{
	}

	template<typename TN, typename TT>
	void set_distribution(TN const & ns, TT const & Ts)
	{
		f_ = [&](coordinates_type const &x )
		{
			return get_value_(ns,x)*inv_sample_density_;
		}
		vs_ = [&](coordinates_type const &x )
		{
			DEFINE_PHYSICAL_CONST
			return std::sqrt(boltzmann_constant * get_value_(Ts,x) / mass_);
		}
	}

private:

	template<typename TF>
	auto get_vaule_(TF const & f, coordinates_type const & x)
	DECL_RET_TYPE((f(x)))

	auto get_vaule_(Real const & f, coordinates_type const & x)
	DECL_RET_TYPE((f))
public:

	void discard(size_t num)
	{
		std::tie(num, std::ignore) = sync_global_location(num * NDIMS * 2);

		rnd_gen.discard(num);
	}

	void set_sample_density(Real sample_density)
	{
		inv_sample_density_ = 1.0 / sample_density;
	}
	void set_cell_id(compact_index_type s)
	{
		s_ = s;
	}
	value_type operator()()
	{

		Vec3 x, v;

		x_dist(rnd_gen, &x[0]);

		v_dist(rnd_gen, &v[0]);

		x = mesh.coordinates_local_to_global(s_, x);

		v *= vs_(x);

		return engine_type::push_forward(std::forward_as_tuple(x, v, f_(x)));
	}

};
}  // namespace simpla

#endif /* PARTICLE_SOURCE_H_ */
