/**
 * @file  mesh.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_MANIFOLD_H_
#define CORE_MESH_STRUCTURED_MANIFOLD_H_
#include <memory>
#include <utility>
#include <vector>
#include <ostream>

#include "../../dataset/dataset.h"
#include "../../utilities/utilities.h"
namespace simpla
{
template<typename ...>struct _Field;

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;

/**
 *  \ingroup manifold
 *  \brief manifold
 */
template<size_t IFORM, //
		typename TG, // Geometric space, mesh
		template<typename > class CalculusPolicy = FiniteDiffMethod, // difference scheme
		template<typename > class InterpolatorPlolicy = InterpolatorLinear // interpolation formula
>
class Manifold
{

public:

	typedef Manifold<IFORM, TG, CalculusPolicy, InterpolatorPlolicy> this_type;

	typedef TG geometry_type;
	typedef typename geometry_type::id_type id_type;
	typedef typename geometry_type::topology_type topology_type;
	typedef typename geometry_type::index_tuple index_tuple;
	typedef CalculusPolicy<geometry_type> calculate_policy;
	typedef InterpolatorPlolicy<geometry_type> interpolatpr_policy;
	typedef typename geometry_type::coordinates_type coordinates_type;
	static constexpr size_t iform = IFORM;
	static constexpr size_t ndims = geometry_type::ndims;
	template<typename TV> using field_value_type=typename
	std::conditional<iform==VERTEX || iform ==VOLUME,TV,nTuple<TV,3>>::type;

	typedef typename geometry_type::template Range<iform> range_type;
	typedef typename range_type::const_iterator const_iterator;

private:
	std::shared_ptr<const geometry_type> m_geometry_;
	range_type m_range_;
public:

	Manifold()
			: m_geometry_(nullptr)
	{
	}

	Manifold(geometry_type const & geo)
			: m_geometry_(geo.shared_from_this()), m_range_()
	{
	}

	Manifold(this_type const & other)
			: m_geometry_(other.m_geometry_)
	{
	}

	~Manifold() = default;

	std::string get_type_as_string() const
	{
		return "Mesh<" + m_geometry_->get_type_as_string() + ">";
	}
	this_type & operator=(this_type const & other)
	{
		m_geometry_ = other.m_geometry_->shared_from_this();
		return *this;
	}

	template<size_t J> using clone_type= Manifold<J,TG, CalculusPolicy, InterpolatorPlolicy>;

	template<size_t J>
	clone_type<J> clone() const
	{
		return clone_type<J>(*m_geometry_);
	}

	/** @name Range Concept
	 * @{
	 */
	template<typename ...Others>
	Manifold(this_type & other, Others && ...others)
			: m_geometry_(other.m_geometry_), m_range_(other.m_range_,
					std::forward<Others>(others)...)
	{
	}
	bool is_divisible() const
	{
		return false; //range_.is_divisible();
	}
	bool empty() const
	{
		return false;
	}

	/**
	 *   @name Geometry
	 *   For For uniform structured grid, the volume of cell is 1.0
	 *   and dx=1.0
	 *   @{
	 */

	void deploy()
	{
	}

	range_type const & range() const
	{
		return m_range_;
	}
	DataSpace dataspace() const
	{
		return std::move(m_geometry_->template dataspace<iform>());
	}

	auto ghost_shape()
	DECL_RET_TYPE(m_geometry_->template ghost_shape<iform>())

	template<typename TFun>
	void serial_foreach(TFun const & fun) const
	{
		for (auto s : m_range_)
		{
			fun(s);
		}
	}

	template<typename TFun, typename ...Args>
	void serial_foreach(TFun const &fun, Args &&...args) const
	{
		for (auto s : m_range_)
		{
			fun(get_value(std::forward<Args>(args),s)...);
		}
	}

	template<typename TFun, typename TContainre>
	void serial_pull_back(TFun const &fun, TContainre & data) const
	{
//		for (auto s : range_)
//		{
//			access(data, s) = sample(fun(geometry_->coordinates(s)), s);
//		}
	}

	template<typename ...Args>
	void pull_back(Args &&...args) const
	{
//		parallel_for(*this, [&](this_type const & sub_m)
//		{
//			sub_m.serial_pull_back(std::forward<Args>(args) ...);
//		});

	}

	/**
	 * @}
	 */

	template<typename ...Args>
	size_t hash(Args && ...args) const
	{
		return m_geometry_->hash(std::forward<Args>(args)...);
	}

	size_t max_hash() const
	{
		return m_geometry_->template max_hash<iform>();
	}
	template<typename ...Args>
	id_type id(Args && ...args) const
	{
		return m_geometry_->id(std::forward<Args>(args)...);
	}

	constexpr id_type id(id_type s) const
	{
		return s;
	}

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE(( calculate_policy::calculate(
							*m_geometry_,std::forward<Args>(args)...)))

	template<typename TOP, typename TL, typename TR>
	void calculate(
			_Field<AssignmentExpression<TOP, TL, TR> > const & fexpr) const
	{
		for (auto s : m_range_)
		{
			fexpr.op_(fexpr.lhs[s],
					calculate_policy::calculate(*m_geometry_, fexpr.rhs, s));
		}
	}

	Real time() const
	{
		return m_geometry_->time();
	}

	template<typename ...Args>
	auto coordinates(Args && ...args) const
	DECL_RET_TYPE(( m_geometry_->coordinates(std::forward<Args>(args)...)))

	template<typename TF>
	auto sample(TF const & v, id_type s) const
	DECL_RET_TYPE(interpolatpr_policy::template sample<iform>(*m_geometry_ ,s,v))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::gather(
							*m_geometry_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::scatter(
							*m_geometry_,std::forward<Args>(args)...)))

//	template<typename ...Args>
//	auto sample(Args && ...args) const
//	DECL_RET_TYPE(
//			(geometry_type:: sample<iform>(
//							*geometry_,std::forward<Args>(args)...)))

};

template<size_t IFORM, typename TG>
std::shared_ptr<Manifold<IFORM, TG>> create_mesh(TG const & geo)
{
	return std::make_shared<Manifold<IFORM, TG>>(geo);
}
template<size_t IFORM, typename TV, typename TG>
_Field<Manifold<IFORM, TG>, std::shared_ptr<TV>> make_form(
		std::shared_ptr<TG> const &geo)
{
	typedef Manifold<IFORM, TG> manifold_type;
	return std::move(
			_Field<manifold_type, std::shared_ptr<TV>>(manifold_type(*geo)));
}
template<size_t IFORM, typename TV, typename TG>
_Field<Manifold<IFORM, TG>, std::shared_ptr<TV>> make_form(TG const & geo)
{
	typedef Manifold<IFORM, TG> manifold_type;
	return std::move(
			_Field<manifold_type, std::shared_ptr<TV>>(manifold_type(geo)));
}

}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_MANIFOLD_H_ */
