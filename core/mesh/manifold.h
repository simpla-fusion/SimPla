/**
 * @file  manifold.h
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

#include "../dataset/dataset.h"
#include "../utilities/utilities.h"
#include "../field/field.h"
#include "mesh_ids.h"

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
		typename CalculusPolicy = FiniteDiffMethod<TG>, // difference scheme
		typename InterpolatorPlolicy = InterpolatorLinear<TG> // interpolation formula
>
class Manifold: public TG
{

public:

	typedef Manifold<IFORM, TG, CalculusPolicy, InterpolatorPlolicy> this_type;

	typedef TG geometry_type;
	typedef typename geometry_type::id_type id_type;
	typedef typename geometry_type::topology_type topology_type;
	typedef typename geometry_type::index_tuple index_tuple;
	typedef typename geometry_type::domain_type domain_type;

	typedef CalculusPolicy calculate_policy;
	typedef InterpolatorPlolicy interpolatpr_policy;

	typedef typename geometry_type::coordinates_type coordinates_type;
	static constexpr size_t iform = IFORM;
	static constexpr size_t ndims = geometry_type::ndims;

private:
	std::shared_ptr<const geometry_type> m_geometry_;
	domain_type m_domain_;
public:

	Manifold() :
			m_geometry_(nullptr)
	{
	}

	Manifold(geometry_type const & geo) :
			m_geometry_(geo.shared_from_this()), m_domain_(
					m_geometry_->domain())
	{
	}

	Manifold(this_type const & other) :
			m_geometry_(other.m_geometry_), m_domain_(m_geometry_->domain())
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
public:
	template<typename TV>
	using field_value_type=typename std::conditional<iform==EDGE ||iform==FACE,nTuple<TV,3>,TV>::type;

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
	Manifold(this_type & other, Others && ...others) :
			m_geometry_(other.m_geometry_)
	{
	}

	/**
	 *   @name Geometry
	 *   For For uniform structured grid, the volume of cell is 1.0
	 *   and dx=1.0
	 *   @{
	 */

	void deploy()
	{
//		ids(m_geometry_->template domain<iform>());
	}

	DataSpace dataspace() const
	{
		return std::move(m_geometry_->template dataspace<iform>());
	}

	auto ghost_shape() const
	DECL_RET_TYPE(m_geometry_->template ghost_shape<iform>())

	auto dx() const
	DECL_RET_TYPE(m_geometry_->dx())
	/**
	 * @}
	 */

	template<typename ...Args>
	size_t hash(Args && ...args) const
	{
		return m_geometry_->template hash<iform>(std::forward<Args>(args)...);
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
//		for (auto s : domain())
//		{
//			fexpr.op_(fexpr.lhs[s],
//					calculate_policy::calculate(*m_geometry_, fexpr.rhs, s));
//		}
	}

	Real time() const
	{
		return m_geometry_->time();
	}

	template<typename ...Args>
	auto coordinates(Args && ...args) const
	DECL_RET_TYPE(( m_geometry_->coordinates(std::forward<Args>(args)...)))

	template<typename TF>
	auto sample(TF const & v,
			id_type s) const
					DECL_RET_TYPE(interpolatpr_policy::template sample<iform>(*m_geometry_ ,s,v))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::gather(
							*m_geometry_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::scatter(
							*m_geometry_,std::forward<Args>(args)...)))

};

template<typename TM, size_t IFORM, typename TV>
using Form=_Field<Manifold<IFORM,TM>,TV, _impl::is_sequence_container >;

template<size_t IFORM, typename TV, typename TM>
_Field<Manifold<IFORM, TM>, TV, _impl::is_sequence_container> make_form(
		TM const & mesh)
{
	return std::move(make_field<TV>(Manifold<IFORM, TM>(mesh)));
}
template<size_t IFORM, typename TV, typename TM>
_Field<Manifold<IFORM, TM>, TV, _impl::is_sequence_container> make_form(
		std::shared_ptr<TM> const & pmesh)
{
	return std::move(make_field<TV>(Manifold<IFORM, TM>(*pmesh)));
}
}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_MANIFOLD_H_ */
