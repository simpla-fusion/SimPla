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
#include "../utilities/utilities.h"
namespace simpla
{

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;

/**
 *  \ingroup manifold
 *  \brief manifold
 */
template<size_t IFORM, //
		typename TG, // Geometric space, mesh
		template<typename > class CalculusPolicy = FiniteDiffMethod, // difference scheme
		template<typename > class InterpolatorPlolicy = InterpolatorLinear, // interpolation formula
		template<typename > class ContainerPolicy = std::shared_ptr>
class Manifold
{

public:

	typedef Manifold<IFORM, TG, CalculusPolicy, InterpolatorPlolicy> this_type;

	typedef TG geometry_type;
	typedef typename geometry_type::id_type id_type;
	typedef CalculusPolicy<geometry_type> calculate_policy;
	typedef InterpolatorPlolicy<geometry_type> interpolatpr_policy;

	static constexpr size_t iform = IFORM;
	template<typename TV> using field_value_type=typename
	std::conditional<iform==VERTEX || iform ==VOLUME,TV,nTuple<TV,3>>::type;

	typedef typename geometry_type::template Range<iform> range_type;
	typedef typename range_type::const_iterator const_iterator;

private:
	typename geometry_type::const_holder geometry_;
	range_type range_;
public:

	Manifold() :
			geometry_(nullptr)
	{
	}

	Manifold(geometry_type const & geo) :
			geometry_(geo.shared_from_this()), range_(
					geo.template range<iform>())
	{
	}

	Manifold(this_type const & other) :
			geometry_(other.geometry_)
	{
	}

	~Manifold() = default;

	std::string get_type_as_string() const
	{
		return "Mesh<" + geometry_->get_type_as_string() + ">";
	}
	this_type & operator=(this_type const & other)
	{
		geometry_ = other.geometry_->shared_from_this();
		return *this;
	}

	template<size_t J> using clone_type= Manifold<J,TG, CalculusPolicy, InterpolatorPlolicy>;

	template<size_t J>
	clone_type<J> clone() const
	{
		return clone_type<J>(*geometry_);
	}

	/** @name Range Concept
	 * @{
	 */

	Manifold(this_type & other, op_split) :
			geometry_(other.geometry_)
//	, range_(other.range_, op_split())
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

	template<typename TFun>
	void serial_foreach(TFun const & fun) const
	{
		for (auto s : range_)
		{
			fun(s);
		}
	}

	template<typename TFun, typename ...Args>
	void serial_foreach(TFun const &fun, Args &&...args) const
	{
		for (auto s : range_)
		{
			fun(get_value(std::forward<Args>(args),s)...);
		}
	}

	template<typename ...Args>
	void foreach(Args &&...args) const
	{
//		parallel_for(*this, [&](this_type const & sub_m)
//		{
		this->serial_foreach(std::forward<Args>(args) ...);
//		});
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
		return geometry_->hash(std::forward<Args>(args)...);
	}

	size_t max_hash() const
	{
		return geometry_->template max_hash<iform>();
	}
	template<typename ...Args>
	id_type id(Args && ...args) const
	{
		return geometry_->id(std::forward<Args>(args)...);
	}

	constexpr id_type id(id_type s) const
	{
		return s;
	}

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE((calculate_policy::calculate(
							*geometry_,std::forward<Args>(args)...)))

	template<typename TOP, typename TL, typename TR>
	void calculate(
			_Field<AssignmentExpression<TOP, TL, TR> > const & fexpr) const
	{
		for (auto s : range_)
		{
			fexpr.op_(fexpr.lhs, fexpr.rhs, s);
		}
	}

	range_type const & range() const
	{
		return range_;
	}
	const_iterator begin() const
	{
		return range_.begin();
	}
	const_iterator end() const
	{
		return range_.end();
	}

	template<typename ...Args>
	auto coordinates(Args && ...args) const
	DECL_RET_TYPE(( geometry_->coordinates(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::gather(
							*geometry_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::scatter(
							*geometry_,std::forward<Args>(args)...)))

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

}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_MANIFOLD_H_ */
