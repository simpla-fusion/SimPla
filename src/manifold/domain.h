/*
 * domain.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

namespace simpla
{
class split_tag;

template<typename TG, unsigned int IFORM>
class Domain
{
private:
	TG const& manifold_;
	std::shared_ptr<Domain<TG, IFORM>> parent_ = nullptr;
public:

//	static constexpr unsigned int ndims = geometry_type::ndims; // number of dimensions of domain D

	static constexpr unsigned int iform = IFORM; // type of form, VERTEX, EDGE, FACE,VOLUME

	typedef TG manifold_type;

	typedef Domain<manifold_type, iform> this_type;

	typedef typename manifold_type::coordinates_type coordinates_type;

	typedef typename manifold_type::index_type index_type;

	Domain(manifold_type const & g) :
			manifold_(g), parent_(nullptr)
	{
	}
	// Copy constructor.
	Domain(const this_type& rhs) :
			manifold_(rhs.manifold_), parent_(rhs.parent_)
	{
	}

	~Domain() = default; // Destructor.

	void swap(this_type& rhs);

	bool empty() const; // True if domain is empty.
	bool is_divisible() const; // True if domain can be partitioned into two subdomains.

	Domain(this_type& d, split_tag); // Split d into two sub-domains.

	typedef size_t difference_type; // Type for difference of two iterators
	struct iterator; // Iterator type for domain

	iterator begin() const; // First item in domain.
	iterator end() const; // One past last item in domain.

	this_type operator &(this_type const & D1) const // \f$D_0 \cap \D_1\f$
	{
		return *this;
	}
	this_type operator |(this_type const & D1) const // \f$D_0 \cup \D_1\f$
	{
		return *this;
	}
	bool operator==(this_type const&)
	{
		return true;
	}
	bool is_same(this_type const&);

	this_type const & parent() const; // Parent domain
	std::tuple<coordinates_type, coordinates_type> boundbox() const; // boundbox on _this_ coordinates system
	std::tuple<nTuple<3, Real>, nTuple<3, Real>> cartesian_boundbox() const; // boundbox on   _Cartesian_ coordinates system
	size_t hash(index_type) const; // get relative  postion of  grid point s in the memory
	size_t max_hash() const; // get max number of grid points in memory

	void for_each(std::function<void(index_type)>)
	{
	}

	template<typename ...Args>
	auto gather(Args &&... args) const DECL_RET_TYPE ((this->manifold_.gather(
							std::integral_constant<unsigned int, iform>(),
							std::forward<Args> (args)...)
			));

	template<typename ...Args>
	auto scatter(Args &&... args) const DECL_RET_TYPE ((this->manifold_.scatter(
							std::integral_constant<unsigned int, iform>(),
							std::forward<Args> (args)...)
			));

	template<typename TOP,typename ...Args>
	auto calculus(Args &&... args) const DECL_RET_TYPE ((this->manifold_.template calculus<TOP>(
							std::forward<Args> (args)...)
			));

};
template<typename TG, unsigned int IFORM>
size_t Domain<TG, IFORM>::hash(index_type s) const
{
	return 0;
}
template<typename TG, unsigned int IFORM>
size_t Domain<TG, IFORM>::max_hash() const
{
	return 0;
}
}
// namespace simpla

#endif /* DOMAIN_H_ */
