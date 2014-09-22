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

template<typename TG, unsigned int IFORM>
class Domain: public TG
{

public:

//	static constexpr unsigned int ndims = geometry_type::ndims; // number of dimensions of domain D

	static constexpr unsigned int iform = IFORM; // type of form, VERTEX, EDGE, FACE,VOLUME

	typedef TG geometry_type;

	typedef Domain<geometry_type, iform> this_type;

	typedef typename geometry_type::coordinates_type coordinates_type;

	typedef typename geometry_type::index_type index_type;

	Domain(const D&); // Copy constructor.
	~Domain(); // Destructor.
	bool empty() const; // True if domain is empty.
	bool is_divisible() const; // True if domain can be partitioned into two subdomains.
	Domain(this_type& d, split); // Split d into two sub-domains.

	typedef size_t difference_type; // Type for difference of two iterators
	struct iterator; // Iterator type for domain

	iterator begin() const; // First item in domain.
	iterator end() const; // One past last item in domain.

	this_type const & parent() const; // Parent domain
	this_type operator &(this_type const & D1) const; // \f$D_0 \cap \D_1\f$
	this_type operator |(this_type const & D1) const; // \f$D_0 \cup \D_1\f$
	bool operator==(this_type const&);
	bool is_same(this_type const&);

	this_type const & parent() const; // Parent domain
	std::tuple<coordinates_type, coordinates_type> boundbox() const; // boundbox on _this_ coordinates system
	std::tuple<nTuple<3, Real>, nTuple<3, Real>> cartesian_boundbox() const; // boundbox on   _Cartesian_ coordinates system
	size_t hash(index_type) const; // get relative  postion of  grid point s in the memory
	size_t max_hash() const; // get max number of grid points in memory

private:
	this_type const& parent_;
};

}  // namespace simpla

#endif /* DOMAIN_H_ */
