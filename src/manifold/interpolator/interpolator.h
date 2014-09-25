/*
 * interpolator.h
 *
 *  created on: 2014-4-17
 *      Author: salmon
 */

#ifndef INTERPOLATOR_H_
#define INTERPOLATOR_H_

namespace simpla
{

template<typename, typename > class Field;
template<typename, unsigned int> class Domain;

/**
 * \ingroup Interpolator
 *
 * \brief basic linear interpolator
 */
template<typename G>
class InterpolatorLinear
{

public:
	typedef InterpolatorLinear<G> this_type;

	G const & geo;
	InterpolatorLinear(G const & g) :
			geo(g)
	{

	}

	InterpolatorLinear(this_type const & r) :
			geo(r.geo)
	{

	}
	~InterpolatorLinear()
	{

	}

//	template<typename ... Args>
//	auto gather(Args && ...args) const
//	DECL_RET_TYPE ((gather_(std::forward<Args> (args)...)))
//
//	template<typename ... Args>
//	auto scatter(Args && ...args) const
//	DECL_RET_TYPE (scatter_(std::forward<Args> (args)...))

private:

	template<typename TD, typename TIDX>
	inline auto gather_impl_(TD const & f,
			TIDX const & idx) const -> decltype(get_value(f, std::get<0>(idx) )* std::get<1>(idx)[0])
	{

		auto X = (G::topology_type::_DI) << 1;
		auto Y = (G::topology_type::_DJ) << 1;
		auto Z = (G::topology_type::_DK) << 1;

		typename G::coordinates_type r = std::get<1>(idx);
		typename G::index_type s = std::get<0>(idx);

		return get_value(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) //
		+ get_value(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) //
		+ get_value(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) //
		+ get_value(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) //
		+ get_value(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) //
		+ get_value(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) //
		+ get_value(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) //
		+ get_value(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:

	template<typename TM,typename TD>
	inline auto gather(Field<Domain<TM, VERTEX>, TD> const &f,
			typename G::coordinates_type const & r) const //
	DECL_RET_TYPE((gather_impl_(f, geo.coordinates_global_to_local(r, 0UL) )))

	template<typename TM,typename TD>
	inline auto gather(
			Field<Domain<TM, EDGE>, TD> const &f,
			typename G::coordinates_type const & r)const
	DECL_RET_TYPE(
			make_ntuple(
					gather_impl_(f, geo.coordinates_global_to_local(r, (G::topology_type::_DI)) ),
					gather_impl_(f, geo.coordinates_global_to_local(r, (G::topology_type::_DJ)) ),
					gather_impl_(f, geo.coordinates_global_to_local(r, (G::topology_type::_DK)) )
			))

	template<typename TM,typename TD>
	inline auto gather (
			Field<Domain<TM, FACE>, TD> const &f,
			typename G::coordinates_type const &r)const
	DECL_RET_TYPE( make_ntuple(

					gather_impl_(f, geo.coordinates_global_to_local(r,((G::topology_type::_DJ | G::topology_type::_DK))) ),
					gather_impl_(f, geo.coordinates_global_to_local(r,((G::topology_type::_DK | G::topology_type::_DI))) ),
					gather_impl_(f, geo.coordinates_global_to_local(r,((G::topology_type::_DI | G::topology_type::_DJ))) )
			) )

	template<typename TM,typename TD>
	inline auto gather (
			Field<Domain<TM, VOLUME>, TD> const &f,
			typename G::coordinates_type const & x)const
	DECL_RET_TYPE(gather_impl_(f, geo.coordinates_global_to_local(x, (G::topology_type::_DA)) ))

private:
	template<typename TD, typename IDX, typename TV>
	inline void scatter_impl_(TD &f, IDX const& idx, TV & v)
	{

		auto X = (G::topology_type::_DI) << 1;
		auto Y = (G::topology_type::_DJ) << 1;
		auto Z = (G::topology_type::_DK) << 1;

		typename G::coordinates_type r = std::get<1>(idx);
		typename G::index_type s = std::get<0>(idx);

		get_value(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
		get_value(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
		get_value(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
		get_value(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
		get_value(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
		get_value(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
		get_value(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
		get_value(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:

	template<typename TM, typename TD, typename TV>
	inline void scatter(Field<Domain<TM, VERTEX>, TD> &f,
			typename G::coordinates_type const &x, TV const & v) const
	{
//		get_value(f, std::get<0>(geo.coordinates_global_to_local_NGP(x, 0UL))) +=
//				v;
		Scatter_impl_(f, geo.coordinates_global_to_local(x, 0UL), v);
	}

	template<typename TM, typename TD, typename TV>
	inline void scatter(Field<Domain<TM, EDGE>, TD> &f,
			typename G::coordinates_type const & x,
			nTuple<3, TV> const & u) const
	{
		scatter_impl_(f,
				geo.coordinates_global_to_local(x, (G::topology_type::_DI)),
				u[0]);
		scatter_impl_(f,
				geo.coordinates_global_to_local(x, (G::topology_type::_DJ)),
				u[1]);
		scatter_impl_(f,
				geo.coordinates_global_to_local(x, (G::topology_type::_DK)),
				u[2]);

	}

	template<typename TM, typename TD, typename TV>
	inline void scatter(Field<Domain<TM, FACE>, TD> &f,
			typename G::coordinates_type const & x,
			nTuple<3, TV> const & u) const
	{

		scatter_impl_(f,
				geo.coordinates_global_to_local(x,
						((G::topology_type::_DJ | G::topology_type::_DK))),
				u[0]);
		scatter_impl_(f,
				geo.coordinates_global_to_local(x,
						((G::topology_type::_DK | G::topology_type::_DI))),
				u[1]);
		scatter_impl_(f,
				geo.coordinates_global_to_local(x,
						((G::topology_type::_DI | G::topology_type::_DJ))),
				u[2]);
	}

	template<typename TM, typename TD, typename TV>
	inline void scatter(Field<Domain<TM, VOLUME>, TD> &f,
			typename G::coordinates_type const & x, TV const & v) const
	{
		scatter_impl_(f,
				geo.coordinates_global_to_local(x, G::topology_type::_DA), v);
	}

}
;

}
// namespace simpla

#endif /* INTERPOLATOR_H_ */
