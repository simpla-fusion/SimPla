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

template<typename TG, size_t IFORM = 0>
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
	std::shared_ptr<const manifold_type> manifold_;
	range_type range_;

public:
	Domain() :
			manifold_(nullptr)
	{
	}
	Domain(std::shared_ptr<const manifold_type> g) :
			manifold_(g->shared_from_this()), range_(
					manifold_->template select<iform>())/*, parent_(*this)*/
	{
	}
	// Copy constructor.
	Domain(const this_type& rhs) :
			manifold_(rhs.manifold_->shared_from_this()), range_(rhs.range_)/*, parent_(rhs.parent_) */
	{
	}
	// Split d into two sub-domains.
	Domain(this_type& d, split_tag) :
			manifold_(d.manifold_->shared_from_this()), range_(d.range_,
					split_tag())
	{
	}

	~Domain() = default; // Destructor.

	bool is_valid() const
	{
		return manifold_ != nullptr;
	}

	void swap(this_type & that)
	{
		sp_swap(manifold_, that.manifold_);
		sp_swap(range_, that.range_);
	}

	template<typename ...Args>
	auto hash(Args && ...args) const
	DECL_RET_TYPE((range_.hash(std::forward<Args>(args)...)))

	auto max_hash() const
	DECL_RET_TYPE((range_.max_hash()))

	auto begin() const
	DECL_RET_TYPE((range_.begin()))

	auto end() const
	DECL_RET_TYPE((range_.end()))

//	auto rbegin() const
//	DECL_RET_TYPE((range_.rbegin()))
//
//	auto rend() const
//	DECL_RET_TYPE((range_.rend()))

	std::shared_ptr<const manifold_type> manifold() const
	{
		return manifold_;
	}

	void manifold(std::shared_ptr<manifold_type> m)
	{
		manifold_ = m;
	}

	// True if domain can be partitioned into two sub-domains.
	bool is_divisible() const
	{
		return range_.is_divisible();
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
//		return manifold_->geometry_type::boundbox<iform>(range_);
//	}
//	std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> cartesian_boundbox() const // boundbox on   _Cartesian_ coordinates system
//	{
//		return manifold_->geometry_type::cartesian_boundbox<iform>(range_);
//	}

	auto dataset_shape() const
	DECL_RET_TYPE(( manifold_->dataset_shape( *this )))
	template<typename ...Args>
	auto dataset_shape(Args &&... args) const
	DECL_RET_TYPE((
					manifold_->dataset_shape(
							*this,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto coordinates(Args && ...args) const
	DECL_RET_TYPE((manifold_->coordinates(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE((manifold_->calculate(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto sample(Args && ... args)
	DECL_RET_TYPE((manifold_->sample(
							std::integral_constant<size_t, iform>(),
							std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((manifold_->gather(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((manifold_->scatter(std::forward<Args>(args)...)))

}
;

template<size_t IFORM, typename TM>
Domain<TM, IFORM> make_domain(std::shared_ptr<TM> const & m)
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
}
// namespace simpla

#endif /* DOMAIN_H_ */
