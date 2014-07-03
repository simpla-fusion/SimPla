/*
 * pic_engine_default.h
 *
 * \date  2013年11月6日
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
 *  @ingroup ParticleEngine
 *  @brief default PIC pusher, using Boris mover
 */
template<typename TM, bool IsImplicit = false, typename Interpolator = typename TM::interpolator_type>
struct PICEngineDefault
{
public:
	enum
	{
		EnableImplicit = IsImplicit
	};
	Real m;
	Real q;

	typedef PICEngineDefault<TM, IsImplicit, Interpolator> this_type;
	typedef TM mesh_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> n_type;

	typedef typename std::conditional<EnableImplicit,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>>,
	        typename mesh_type:: template field<EDGE, scalar_type> >::type J_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
	};

	typedef std::tuple<coordinates_type, Vec3, Real> compact_point_type;

	auto Compact(Point_s & p) DECL_RET_TYPE(std::tie(p.x,p.v,p.f))
	auto Compact(Point_s && p) DECL_RET_TYPE(std::make_tuple(p.x,p.v,p.f))
	auto Decompact(compact_point_type && p) DECL_RET_TYPE(Point_s(
					{	std::get<0>(p),std::get<1>(p),std::get<2>(p)}))

private:
	Real cmr_;
public:
	mesh_type const &mesh;

	PICEngineDefault(mesh_type const &m)
			: mesh(m), m(1.0), q(1.0), cmr_(1.0)
	{
	}
	template<typename ...Others>
	PICEngineDefault(mesh_type const &pmesh, Others && ...others)
			: PICEngineDefault(pmesh)
	{
		Load(std::forward<Others >(others)...);
	}
	template<typename TDict, typename ...Args>
	void Load(TDict const& dict, Args const & ...args)
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

	static std::string GetTypeAsString()
	{
		return "Default";
	}

	std::string Save(std::string const & path = "", bool is_verbose = false) const
	{
		std::stringstream os;

		DEFINE_PHYSICAL_CONST
		;

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		;
		return os.str();

	}
	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TJ, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB,
			Others const &...others) const
	{
		NextTimeStepZero(Bool2Type<EnableImplicit>(), p, dt, J, fE, fB);
	}
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Point_s * p, Real dt, TE const &fE, TB const & fB, Others const &...others) const
	{
		NextTimeStepHalf(Bool2Type<EnableImplicit>(), p, dt, fE, fB);
	}
// x(-1/2->1/2),v(0)
	template<typename TJ, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Bool2Type<true>, Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB,
			Others const &...others) const
	{
		//		auto B = interpolator_type::Gather(fB, p->x);
		//		auto E = interpolator_type::Gather(fE, p->x);

		p->x += p->v * dt;

		interpolator_type::ScatterCartesian(J,std::make_tuple(p->x,p-> v) ,p->f * q);
	}
// v(0->1)
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Bool2Type<true>, Point_s * p, Real dt, TE const &fE, TB const & fB,
			Others const &...others) const
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
// x(-1/2->1/2), v(-1/2/1/2)
	template<typename TJ, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Bool2Type<false>, Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB,
			Others const &...others) const
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

		interpolator_type::ScatterCartesian(J, p->x, p->v ,p->f * q);

	}
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Bool2Type<false>, Point_s * p, Real dt, TE const &fE, TB const & fB,
			Others const &...others) const
	{
	}

	template<int IFORM, typename TV, typename ...Args>
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
operator<<(OS& os, typename PICEngineDefault<TM...>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}

}
// namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
