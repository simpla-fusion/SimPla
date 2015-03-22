/**
 * @file  domain.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

#include <memory>
#include <type_traits>

#include "../data_interface/data_space.h"
#include "../utilities/type_traits.h"
#include "../design_pattern/expression_template.h"
namespace simpla
{
/** @ingroup  diff_geo
 *  @addtogroup domain Domain
 *
 * ## Summary
 *
 *>  "In mathematics, the  domain of definition or simply the domain of
 *>  a function is the set of "input" or argument values for which the
 *>  function is defined. That is, the function provides an "output"
 *>  or value for each member of the domain.  Conversely, the set of
 *>  values the function takes is termed the image of the function,
 *>  which is sometimes also referred to as the  range of the function."
 *>
 *>  -- [Wiki:Domain of a function](http://en.wikipedia.org/wiki/Domain_of_a_function)
 *
 * - Domain is a [TBB parallel range](https://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/container_range_concept.htm)
 *
 *> "A Range can be recursively subdivided into two parts. It is
 *> recommended that the division be into nearly equal parts, but it
 *> is not required. Splitting as evenly as possible typically yields
 *> the best parallelism. Ideally, a range is recursively splittable
 *>  until the parts represent portions of work that are more efficient
 *>  to execute serially rather than split further. The amount of work
 *>   represented by a Range typically depends upon higher level context,
 *>   hence a typical type that models a Range should provide a way to
 *>   control the degree of splitting. For example, the template class
 *>   blocked_range has a grainsize parameter that specifies the biggest
 *>    range considered indivisible."
 *>
 *>     -- [TBB:Range concept](https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/range_concept.htm)
 *
 * - Domain is a chain of simplex or polytope
 *
 *> In geometry, a simplex (plural simplexes or simplices) is a
 *>  generalization of the notion of a triangle or tetrahedron to
 *>   arbitrary dimensions. Specifically, a k-simplex is a k-dimensional
 *>   polytope which is the convex hull of its k + 1 vertices. More formally,
 *>   suppose the k + 1 points \f$ u_0,\dots, u_k \in \mathbb{R}^n \f$
 *>   are affinely independent, which means \f$ u_1 - u_0,\dots, u_k-u_0 \f$
 *>   are linearly independent. Then, the simplex determined by them
 *>   is the set of points  \f$    C =\{\theta_0 u_0 + \dots+\theta_k u_k | \theta_i \ge 0, 0 \le i \le k, \sum_{i=0}^{k} \theta_i=1\}  \f$.
 *>   -- Wiki
 *
 * ## Requirement
 * The following table lists requirements for  Domain type D as _parallel range_ , which are same as Container Range in TBB.
 *
 * Pseudo-Signature  				| Semantics
 * ---------------------------------|--------------
 * D( const D& ) 					| Copy constructor.
 * ~D() 							| Destructor.
 * bool empty() const 				| True if domain is empty.
 * bool is_divisible() const 		| True if domain can be partitioned into two subdomains.
 * D( D& d, split ) 				| Split d into two subdomains.
 * difference_type 					| Type for difference of two iterators
 * iterator 						| Iterator type for domain
 * iterator begin(  ) const			| First item in domain.
 * iterator end(  ) const 			| One past last item in domain.
 *
 *
 *
 *
 * Additional requirements for  Domain type D
 *
 * Pseudo-Signature  					| Semantics
 * -------------------------------------|-------------
 * D const & parent()const				| Parent domain
 * D operator &(D const & D1)const		| \f$ D_0 \cap  D_1 \f$
 * bool operator==(D const & D1)const	| \f$ D_0 ==  D_1 \f$
 * bool is_same(D const & D1)const		| \f$ D_0 ==  D_1 \f$
 * D operator \|(D const & D1)const		| \f$ D_0 \cup  D_1\f$
 *
 * Requirements for  Domain type D as a geometric object, which could be a @ref concept_simplex or a chain of polytopes.
 *
 * Pseudo-Signature  				| Semantics
 * ---------------------------------|-------------
 * unsigned int iform				| type of form, VERTEX, EDGE, FACE,VOLUME
 * geometry_typ						| Geometry
 * PD boundary(  D const& )			| Boundary of domain D, PD::ndims=D::ndims-1.
 * D const & parent()const			| Parent domain
 * boundbox() const					| boundbox on _this_ coordinates system
 * cartesian_boundbox() const		| boundbox on _Cartesian_ coordinates system
 * size_t hash(index_type)const 	| get relative  postion of  grid point s in the memory
 * size_t max_hash( )const 			| get max number of grid points in memory
 *
 *
 *
 *
 * Pseudo-Signature  							| Semantics
 * ---------------------------------------------|-------------
 * gather(coordinates_type x,TF const& f)const 	| get value at x
 * scatter(coordiantes_type x,v,TF f *  )const 	| scatter v at x to f
 *
 */

template<typename TG, size_t IFORM = 0>
class Domain: public TG, public enable_create_from_this<Domain<TG, IFORM>>
{

public:
	typedef TG manifold_type;

	static constexpr size_t ndims = manifold_type::ndims; // number of dimensions of domain D

	static constexpr size_t iform = IFORM; // type of form, VERTEX, EDGE, FACE,VOLUME

	typedef Domain<manifold_type, iform> this_type;

	typedef Domain<manifold_type, iform> domain_type;

//	typedef typename manifold_type::topology_type topology_type;
//
//	typedef typename manifold_type::coordinates_type coordinates_type;
//
//	typedef typename manifold_type::id_type id_type;
//
//	typedef size_t difference_type; // Type for difference of two iterators
//
//	typedef typename manifold_type::range_type range_type;
//
//	typedef typename range_type::iterator iterator;

private:

	range_type range_;

public:

	Domain(manifold_type const & g) :
			manifold_(g), range_(manifold_type::template select<iform>())/*, parent_(*this)*/
	{
	}
	/// Split d into two sub-domains
	Domain(this_type& rhs, op_split) :
			manifold_(rhs), range_(rhs.range_, op_split())/*, parent_(rhs.parent_) */
	{
	}

	virtual ~Domain() = default; // Destructor.

	template<typename ...Args>
	auto hash(Args && ...args) const DECL_RET_TYPE((range_.hash(std::forward<Args>(args)...)))

	auto max_hash() const
	DECL_RET_TYPE((range_.max_hash()))

//	auto rbegin() const
//	DECL_RET_TYPE((range_.rbegin()))
//
//	auto rend() const
//	DECL_RET_TYPE((range_.rend()))

	// True if domain can be partitioned into two sub-domains.

	bool empty()const
	{
		return size()==0;
	}
	bool is_divisible() const
	{
		return range_.is_divisible();
	}

	size_t size() const
	{
		return range_.size();
	}

	template<typename TV> using container_type=std::shared_ptr<TV>;

	template<typename TV>
	container_type<TV> allocate() const
	{
		return sp_make_shared_array<TV>(size());
	}

	template<typename TV>
	DataSet dataset(container_type<TV> const& data_,
			Properties const& prop) const
	{
		return DataSet( { data_, make_datatype<TV>(), dataspace(), prop });
	}

	template<typename TV>
	auto index_value(container_type<TV> & v, id_type s) const DECL_RET_TYPE(v.get()[s])

	template<typename TV>
	auto index_value (TV* v,id_type s)const
	DECL_RET_TYPE(v[s])

	template<typename TV>
	auto index_value (TV const & v,id_type s)const
	DECL_RET_TYPE(get_value(v,s))

	template<typename ...Args>
	auto sample(Args && ...args)const
	DECL_RET_TYPE((manifold_type::template sample<iform>(std::forward<Args>(args)...)))

	template<typename TOP,typename T1,typename ...Args>
	void foreach(T1 && data,TOP const & op, Args && ... args)const
	{
		for(auto s:range_)
		{
			op(get_value(std::forward<T1>(data),hash(s)), get_value(std::forward<Args>(args),s)...);
		}
	}
	template<typename T1, typename TFun>
	void pull_back(T1 & data, TFun const &fun) const
	{
		for (auto s : range_)
		{
			//FIXME geometry coordinates convert
			get_value(data, hash(s)) = sample(s, fun(coordinates(s)));
		}
	}

}
;

template<size_t IFORM, typename TM>
Domain<TM, IFORM> make_domain(TM const & m)
{
	return std::move(Domain<TM, IFORM>(m));
}
template<size_t IFORM, typename TM>
Domain<TM, IFORM> make_domain()
{
	auto m = std::make_shared<TM>();
	return std::move(Domain<TM, IFORM>(m));
}
template<size_t IFORM, typename TM, typename ...Args>
Domain<TM, IFORM> make_domain(Args && ...args)
{
	auto m = std::make_shared<TM>(std::forward<Args>(args)...);
	return std::move(Domain<TM, IFORM>(m));
}

template<typename T> struct domain_traits
{
	typedef std::nullptr_t manifold_type;
	static constexpr size_t iform = VERTEX;
};

template<typename TM, size_t IFORM>
struct domain_traits<Domain<TM, IFORM> >
{
	typedef TM manifold_type;
	static constexpr size_t iform = IFORM;
	typedef typename manifold_type::coordinates_type coordinates_type;
	typedef typename manifold_type::index_type index_type;
};

}
// namespace simpla

#endif /* DOMAIN_H_ */
