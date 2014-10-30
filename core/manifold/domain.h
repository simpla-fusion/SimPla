/*
 * domain.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_
#include "../utilities/sp_type_traits.h"

namespace simpla
{
class split_tag;
template<typename ...> class _Field;

template<typename TG, size_t IFORM>
class Domain
{

public:
	typedef TG manifold_type;

	static constexpr size_t ndims = manifold_type::ndims; // number of dimensions of domain D

	static constexpr size_t iform = IFORM; // type of form, VERTEX, EDGE, FACE,VOLUME

	typedef Domain<manifold_type, iform> this_type;

	typedef Domain<manifold_type, iform> domain_type;

	typedef typename manifold_type::topology_type topology_type;

	typedef typename manifold_type::coordinates_type coordinates_type;

	typedef typename manifold_type::index_type index_type;

	typedef size_t difference_type; // Type for difference of two iterators

	typedef typename manifold_type::range_type range_type;

	typedef typename range_type::iterator iterator;

public:
	manifold_type const& manifold_;
	range_type range_;
//private:
//	Domain<TG, IFORM> const & parent_;

public:
	Domain(manifold_type const & g) :
			range_(g.select(iform)), manifold_(g)/*, parent_(*this)*/
	{
	}
	// Copy constructor.
	Domain(const this_type& rhs) :
			range_(rhs.range_), manifold_(rhs.manifold_)/*, parent_(rhs.parent_) */
	{
	}
//	Domain(this_type& d, split_tag); // Split d into two sub-domains.

	~Domain() = default; // Destructor.

	template<typename ...Args>
	auto hash(Args && ...args) const
	DECL_RET_TYPE((range_. hash(std::forward<Args>(args)...)))

	auto max_hash() const
	DECL_RET_TYPE((range_.max_hash()))

	auto begin() const
	DECL_RET_TYPE((range_.begin()))

	auto end() const
	DECL_RET_TYPE((range_.end()))

	auto rbegin() const
	DECL_RET_TYPE((range_.rbegin()))

	auto rend() const
	DECL_RET_TYPE((range_.rend()))

	void swap(this_type& rhs);

	manifold_type const & manifold() const
	{
		return manifold_;
	}

	// True if domain can be partitioned into two sub-domains.
	bool is_divisible() const
	{
		return false;
	}

//	this_type operator &(this_type const & D1) const // \f$D_0 \cap \D_1\f$
//	{
//		return *this;
//	}
//	this_type operator |(this_type const & D1) const // \f$D_0 \cup \D_1\f$
//	{
//		return *this;
//	}
//	bool operator==(this_type const&)
//	{
//		return true;
//	}
//	bool is_same(this_type const&);
//
//	this_type const & parent() const; // Parent domain
//
//	std::tuple<coordinates_type, coordinates_type> boundbox() const // boundbox on _this_ coordinates system
//	{
//		return manifold_.geometry_type::boundbox<iform>(range_);
//	}
//	std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> cartesian_boundbox() const // boundbox on   _Cartesian_ coordinates system
//	{
//		return manifold_.geometry_type::cartesian_boundbox<iform>(range_);
//	}

	auto dataset_shape() const
	DECL_RET_TYPE(( manifold_.get_dataset_shape( *this )))
	template<typename ...Args>
	auto dataset_shape(Args &&... args) const
	DECL_RET_TYPE((
					manifold_.get_dataset_shape(
							*this,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto sample(Args && ... args)
	DECL_RET_TYPE((manifold_.sample(
							std::integral_constant<size_t, iform>(),
							std::forward<Args>(args)...)))

	template<typename ...Args>
	auto calculate(Args && ...args)const
	DECL_RET_TYPE((manifold_._get_value(std::forward<Args>(args)...)))

public:
	template<typename TL, typename TR>
	void assign(TL & lhs, TR const & rhs) const
	{
		parallel_for(get_domain(lhs) & get_domain(rhs),

		[& ](this_type const &r)
		{
			for(auto const & s:r)
			{
				lhs[s]= manifold_.get_value( rhs,s );
			}
		});

	}

}
;

template<size_t IFORM, typename TM>
Domain<TM, IFORM> make_domain(TM const & m)
{
	return std::move(Domain<TM, IFORM>(m));
}

template<typename TG, size_t IFORM, typename ...Args>
auto calculate(Domain<TG, IFORM> const & d, Args && ... args)
DECL_RET_TYPE((d.manifold().calculate(std::forward<Args>(args)...)))

template<typename TG, size_t IFORM, typename ...Args>
auto gather(Domain<TG, IFORM> const & d, Args && ... args)
DECL_RET_TYPE((d.manifold().gather(std::forward<Args>(args)...)))

template<typename TG, size_t IFORM, typename ...Args>
auto scatter(Domain<TG, IFORM> const & d, Args && ... args)
DECL_RET_TYPE((d.manifold().scatter(std::forward<Args>(args)...)))
}
// namespace simpla

#endif /* DOMAIN_H_ */
