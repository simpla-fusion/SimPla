/*
 * pic_engine_default.h
 *
 * \date  2013-11-6
 *      \author  salmon
 */

#ifndef PIC_ENGINE_DEFAULT_H_
#define PIC_ENGINE_DEFAULT_H_

#include <sstream>
#include <string>
#include <type_traits>

#include "../../src/fetl/fetl.h"
#include "../../src/physics/physical_constants.h"
#include "../../src/utilities/sp_type_traits.h"
#include "../../src/io/hdf5_datatype.h"

namespace simpla
{

/**
 *  \ingroup ParticleEngine
 *  \brief default PIC pusher, using Boris mover
 */
template<typename TM, typename Interpolator = typename TM::interpolator_type>
struct PICEngineDefault
{
public:
	enum
	{
		is_implicit = false
	};
	Real m;
	Real q;

	typedef PICEngineDefault<TM, Interpolator> this_type;
	typedef TM mesh_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> n_type;

	typedef typename mesh_type:: template field<EDGE, scalar_type> J_type;

	typedef typename mesh_type:: template field<EDGE, scalar_type> E_type;

	typedef typename mesh_type:: template field<FACE, scalar_type> B_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;

		typedef std::tuple<coordinates_type, Vec3, Real> compact_type;

		static compact_type Compact(Point_s const& p)
		{
			return ((std::make_tuple(p.x, p.v, p.f)));
		}

		static Point_s Decompact(compact_type const & t)
		{
			Point_s p;
			p.x = std::get<0>(t);
			p.v = std::get<1>(t);
			p.f = std::get<2>(t);
			return std::move(p);
		}
	};

	typedef std::tuple<coordinates_type, Vec3, Real> compact_point_type;

private:
	Real cmr_;
public:
	mesh_type const &mesh;

	PICEngineDefault(mesh_type const &m) :
			mesh(m), m(1.0), q(1.0), cmr_(1.0)
	{
	}
	template<typename ...Others>
	PICEngineDefault(mesh_type const &pmesh, Others && ...others) :
			PICEngineDefault(pmesh)
	{
		load(std::forward<Others >(others)...);
	}
	template<typename TDict, typename ...Args>
	void load(TDict const& dict, Args const & ...args)
	{
		m = (dict["Mass"].template as<Real>(1.0));
		q = (dict["Charge"].template as<Real>(1.0));

		cmr_ = (q / m);
		{
			std::ostringstream os;
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "}";

			GLOBAL_HDF5_DATA_TYPE_FACTORY.template Register < Point_s > (os.str());
		}

	}

	~PICEngineDefault()
	{
	}

	static std::string get_type_as_string()
	{
		return "Default";
	}

	Real get_mass()const
	{
		return m;

	}
	Real get_charge()const
	{
		return q;

	}
	template<typename OS>
	OS & print(OS & os) const
	{

		DEFINE_PHYSICAL_CONST
		;

		os << "Engine = '" << get_type_as_string() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		;
		return os;

	}

	// x(-1/2->1/2), v(-1/2/1/2)
	inline void next_timestep_zero( Point_s * p, Real dt, E_type const &fE, B_type const & fB ) const
	{

		p->x += p->v * dt * 0.5;
		auto B = real(interpolator_type::GatherCartesian(fB, p->x));
		auto E = real(interpolator_type::GatherCartesian(fE, p->x));

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_ * 2.0;

		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

	}

	inline void next_timestep_half( Point_s * p, Real dt, E_type const &fE, B_type const & fB ) const
	{
	}

	void Scatter(Point_s const & p, J_type * J ) const
	{
		interpolator_type::ScatterCartesian( J,std::make_tuple(p.x,p.v), p.f * q);
	}

	void Scatter(Point_s const & p, n_type * n) const
	{
		interpolator_type::ScatterCartesian( n,std::make_tuple(p.x,1.0),p.f * q);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s(
						{	x, v, f}));
	}

};
template<typename OS, typename ... TM> OS&
operator<<(OS& os, typename PICEngineDefault<TM...>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}
}
// namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
