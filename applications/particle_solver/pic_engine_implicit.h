/*
 * pic_engine_implicit.h
 *
 * \date  2014-4-10
 *      \author  salmon
 */

#ifndef PIC_ENGINE_IMPLICIT_H_
#define PIC_ENGINE_IMPLICIT_H_

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
struct PICEngineImplicit
{
public:
	enum
	{
		is_implicit = true
	};
	Real m;
	Real q;

	typedef PICEngineImplicit<TM, Interpolator> this_type;
	typedef TM mesh_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> n_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> J_type;

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

	PICEngineImplicit(mesh_type const &m) :
			mesh(m), m(1.0), q(1.0), cmr_(1.0)
	{
	}
	template<typename ...Others>
	PICEngineImplicit(mesh_type const &pmesh, Others && ...others) :
			PICEngineImplicit(pmesh)
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

	~PICEngineImplicit()
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
	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

// x(-1/2->1/2),v(0)
	template<typename TE, typename TB >
	inline void next_timestep_zero( Point_s * p, Real dt , TE const &fE, TB const & fB ) const
	{
		p->x += p->v * dt;
	}
	template< typename TE, typename TB,typename TJ >
	inline void next_timestep_zero(Point_s * p, Real dt, TE const &fE, TB const & fB, TJ *J ) const
	{
		next_timestep_zero(p,dt,fE,fB);
		interpolator_type::ScatterCartesian( J,std::make_tuple(p->x,p->v), p->f * q);
	}
// v(0->1)
	template<typename TE, typename TB >
	inline void next_timestep_half( Point_s * p, Real dt, TE const &fE, TB const & fB ) const
	{

		auto B = real(interpolator_type::GatherCartesian(fB, p->x));
		auto E = real(interpolator_type::GatherCartesian(fE, p->x));

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_ * 2.0;

		p->v += E * (cmr_ * dt * 0.5);

	}

	template<typename TE, typename TB,typename TJ>
	inline void next_timestep_half(Point_s * p, Real dt, TE const &fE, TB const & fB,TJ* J ) const
	{
	}

	template<unsigned int IFORM, typename TV, typename ...Args>
	void Scatter(Point_s const & p, typename mesh_type:: template field < IFORM, TV> * n, Args const & ...) const
	{
		interpolator_type::ScatterCartesian( n, p.x, q * p.f);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s(
						{	x, v, f}));
	}

};
template<typename OS, typename ... TM> OS&
operator<<(OS& os, typename PICEngineImplicit<TM...>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}
}
// namespace simpla

#endif /* PIC_ENGINE_IMPLICIT_H_ */
