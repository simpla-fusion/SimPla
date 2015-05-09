/*
 * @file sp_sorted_set.h
 *
 *  created on: 2014-4-29
 *      Author: salmon
 */

#ifndef CORE_GTL_CONTAINER_SORTED_SET_H_
#define CORE_GTL_CONTAINER_SORTED_SET_H_

#include <stddef.h>
#include <forward_list>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include "../../dataset/dataset.h"
#include "../iterator/sp_indirect_iterator.h"
namespace simpla
{
/**
 * @brief an alternative implement of `std::unordered_multiset`
 * - elements are organized into buckets,
 * - optimize for frequently rehash,insert and remove
 * -
 *
 *>  Unordered multiset is an associative container that contains set
 *> of possibly non-unique objects of type Key. --http://en.cppreference.com/w/cpp/container/unordered_multiset
 */
template<typename T, typename Hash = std::hash<T>,
		typename Allocator = std::allocator<T> >
class sp_sorted_set
{

public:
	typedef T value_type;
	typedef Hash hasher;
	typedef Allocator allocator_type;
	typedef typename std::result_of<hasher(value_type const &)>::type key_type;

	typedef sp_sorted_set<value_type, hasher, allocator_type> this_type;

	typedef std::forward_list<T, allocator_type> bucket_type;

	typedef std::map<key_type, bucket_type> base_container_type;

//private:

	hasher m_hasher_;

	base_container_type m_data_;

	std::set<key_type> m_modified_;

public:

	// Constructor
	sp_sorted_set()
	{
	}
	sp_sorted_set(this_type const & other) :
			m_hasher_(other.m_hasher_), m_data_(other.m_data_)
	{
	}

	sp_sorted_set(this_type && other) :
			m_hasher_(other.m_hasher_), m_data_(other.m_data_)
	{
	}

	template<typename TRange>
	sp_sorted_set(this_type & other, TRange const &range)
	{
		splice(range, other);
	}

	~sp_sorted_set()
	{
	}

	hasher const & hash_function() const
	{
		return m_hasher_;
	}

	void hash_function(hasher const & h_fun)
	{
		m_hasher_ = h_fun;
	}

	void swap(base_container_type & other)
	{
		m_data_.swap(other);
	}

	void swap(this_type & other)
	{
		m_data_.swap(other.m_data_);
		std::swap(m_hasher_, other.m_hasher_);
	}

	bool empty() const
	{
		return m_data_.empty();
	}
	bool is_divisible() const
	{
		return false;
	}
	bucket_type & operator[](key_type const & key)
	{
		m_modified_.insert(key);
		return m_data_[key];
	}

	bucket_type & at(key_type const & key)
	{
		m_modified_.insert(key);
		return m_data_.at(key);
	}

	bucket_type const& at(key_type const & key) const
	{
		return m_data_.at(key);
	}

	typedef typename bucket_type::iterator local_iterator;
	typedef typename bucket_type::const_iterator const_local_iterator;

	local_iterator begin(key_type const & id)
	{
		m_modified_.insert(id);
		return std::move(m_data_[id].begin());
	}

	local_iterator end(key_type const & id)
	{
		return std::move(m_data_[id].end());
	}

	constexpr const_local_iterator cbegin(key_type const & id) const
	{
		return std::move(m_data_.at(id).begin());
	}

	constexpr const_local_iterator cend(key_type const & id) const
	{
		return std::move(m_data_.at(id).end());
	}

	constexpr const_local_iterator begin(key_type const & id) const
	{
		return std::move(m_data_.at(id).begin());
	}

	constexpr const_local_iterator end(key_type const & id) const
	{
		return std::move(m_data_.at(id).end());
	}

	template<typename TV>
	void insert(key_type const & key, TV && v)
	{
		(*this)[key].insert_after(m_data_[key].before_begin(),
				std::forward<TV>(v));
	}

	template<typename TV>
	key_type insert(TV const& v)
	{
		auto s = m_hasher_(v);

		(*this)[s].push_front(v);

		return s;
	}

	template<typename TV>
	void push_front(TV && v)
	{
		(*this)[m_hasher_(v)].push_front(std::forward<TV>(v));
	}

	void insert(std::initializer_list<value_type> ilist)
	{
		insert(ilist.begin(), ilist.end());
	}

	template<typename InputIter>
	void insert(InputIter first, InputIter last)
	{
		for (auto it = first; it != last; ++it)
		{
			insert(*it);
		}
	}

	void insert(key_type const & key, std::initializer_list<value_type> ilist)
	{
		(*this)[key].insert_after(m_data_[key].before_begin(), ilist);
	}

	template<typename InputIter>
	void insert(key_type const & key, InputIter first, InputIter last)
	{
		(*this)[key].insert_after(m_data_[key].before_begin(), first, last);
	}

	template<typename ...Others>
	void assign(key_type const & key, Others && ...others)
	{
		(*this)[key].assign(std::forward<Others>(others)...);
	}

	template<typename IputIterator>
	void assign(IputIterator first, IputIterator last)
	{
		clear();
		insert(first, last);
	}

	template<typename TRange>
	void splice(TRange const & range, this_type & other)
	{
		for (auto const & key : range)
		{
			auto it = other.m_data_.find(key);
			if (it != other.m_data_.end())
			{
				auto & dest = (*this)[it->first];

				dest.splice_after(dest.before_begin(), it->seond);
			}
		}
	}
	/**
	 *  BucketInputIter=  std::pair<key_type,bucket_type> *
	 * @param first
	 * @param last
	 */
	template<typename BucketInputIter>
	void splice(BucketInputIter first, BucketInputIter last)
	{
		for (auto it = first; it != last; ++it)
		{
			auto & dest = (*this)[it->first];

			dest.splice_after(dest.before_begin(), it->second);
		}
	}

	void erase(key_type const & key)
	{
		m_data_.erase(key);
		m_modified_.erase(key);

	}
	void erase_all()
	{
		m_data_.clear();
		m_modified_.clear();
	}
	void clear()
	{
		erase_all();

	}
	template<typename TRange>
	void erase(TRange const & range)
	{
		for (auto const & key : range)
		{
			erase((key));
		}

	}
	/**
	 *  move  elements  which `hash(value)!=key`  from   `m_data_[key]`
	 *   to container `other[hash(value)]`
	 * @param key
	 * @param other
	 * @return number of moved elements
	 */
private:
	size_t rehash_one(typename base_container_type::iterator & it,
			base_container_type & other)
	{
		if (it == m_data_.end())
		{
			return 0;
		}
		auto const & key = it.first;
		auto & bucket = it.second;
		auto pt = bucket.begin();
		size_t count = 0;
		while (pt != bucket.end())
		{
			auto p = pt;
			++pt;

			auto o_key = m_hasher_(*p);

			if (o_key != key)
			{
				auto & dest = other[o_key];
				dest.splice_after(dest.before_begin(), bucket, p);
				++count;
			}
		}
		m_modified_.erase(key);

		return count;
	}
public:
	template<typename TRange>
	void rehash(TRange const & r)
	{
		base_container_type other;

		for (auto const & s : r)
		{
			if (m_modified_.find(s) != m_modified_.end())
				rehash_one(m_data_.find(s), other);
		}

		splice(other.begin(), other.end());

		VERBOSE << " Rehash/Resort particles" << endl;

	}

	void rehash()
	{
		base_container_type other;

		for (auto s : m_modified_)
		{
			rehash_one(m_data_.find(s), other);
		}

		splice(other.begin(), other.end());

		VERBOSE << " Rehash/Resort particles" << endl;

	}

	size_t size(key_type const & key) const
	{
		size_t count = 0;

		auto item = m_data_.find(key);
		if (item != m_data_.end())
		{
			count = std::distance(item->second.begin(), item->second.end());
		}
		return count;
	}
	template<typename TRange>
	size_t size_all(TRange const & range) const
	{
		size_t count = 0;
		for (auto const & key : range)
		{
			count += size(key);
		}
		return count;
	}
	size_t size() const
	{

		size_t count = 0;
		for (auto const & item : m_data_)
		{
			count += std::distance(item.second.begin(), item.second.end());
		}
		return count;
	}

	typedef typename base_container_type::iterator iterator;
	typedef typename base_container_type::const_iterator const_iterator;

	iterator begin()
	{
		return m_data_.begin();
	}
	iterator end()
	{
		return m_data_.end();
	}

	const_iterator begin() const
	{
		return m_data_.cbegin();
	}
	const_iterator end() const
	{
		return m_data_.cend();
	}
	const_iterator cbegin() const
	{
		return m_data_.cbegin();
	}
	const_iterator cend() const
	{
		return m_data_.cend();
	}

	auto find(key_type const & key)
	DECL_RET_TYPE(this->m_data_.find(key))

	auto find(key_type const & key) const
	DECL_RET_TYPE(this->m_data_.find(key))

	std::list<std::reference_wrapper<bucket_type>> select()
	{
		std::list<std::reference_wrapper<bucket_type>> res;
		for (auto const & it : m_data_)
		{
			res.push_back(std::ref(it->second));
		}
		return std::move(res);
	}

	std::list<std::reference_wrapper<const bucket_type>> select() const
	{
		std::list<std::reference_wrapper<const bucket_type>> res;
		for (auto const & it : m_data_)
		{
			res.push_back(std::cref(it->second));
		}
		return std::move(res);
	}

//	template<typename TRange>
//	std::list<std::reference_wrapper<bucket_type>> select(TRange const & xrange)
//	{
//		std::list<std::reference_wrapper<bucket_type>> res;
//		for (auto const & id : xrange)
//		{
//			auto it = m_data_.find(id);
//			if (it != m_data_.end())
//			{
//				res.push_back(std::ref(it->second));
//			}
//		}
//		return std::move(res);
//	}
//	template<typename TRange>
//	std::list<std::reference_wrapper<const bucket_type>> select(
//			TRange const & xrange) const
//	{
//		std::list<std::reference_wrapper<const bucket_type>> res;
//
//		for (auto const & id : xrange)
//		{
//			auto it = m_data_.find(id);
//			if (it != m_data_.end())
//			{
//				res.push_back(std::cref(it->second));
//			}
//		}
//		return std::move(res);
//	}

}
;

}
// namespace simpla

//private:
//
//	template<bool IsConst>
//	struct cell_iterator_
//	{
//
//		typedef cell_iterator_<IsConst> this_type;
//
//		/// One of the @link iterator_tags tag types@endlink.
//		typedef std::forward_iterator_tag iterator_category;
//
//		/// The type "pointed to" by the iterator.
//		typedef typename std::conditional<IsConst, const cell_type, cell_type>::type value_type;
//
//		/// This type represents a pointer-to-value_type.
//		typedef value_type * pointer;
//
//		/// This type represents a reference-to-value_type.
//		typedef value_type & reference;
//
//		typedef typename std::conditional<IsConst, container_type const &, container_type&>::type container_reference;
//
//		typedef typename std::conditional<IsConst, typename container_type::const_iterator,
//				typename container_type::iterator>::type cell_iterator;
//
//		container_reference data_;
//		key_type m_it_, m_ie_;
//		cell_iterator c_it_;
//
//		cell_iterator_(container_reference data, mesh_iterator m_ib, mesh_iterator m_ie) :
//				data_(data), m_it_(m_ib), m_ie_(m_ie), c_it_(base_container_type::find(m_ib))
//		{
//			UpldateCellIteraor_();
//		}
//
//		~cell_iterator_()
//		{
//		}
//
//		reference operator*()
//		{
//			return c_it_->second;
//		}
//		const reference operator*() const
//		{
//			return c_it_->second;
//		}
//		pointer operator ->()
//		{
//			return &(c_it_->second);
//		}
//		const pointer operator ->() const
//		{
//			return &(c_it_->second);
//		}
//		bool operator==(this_type const & rhs) const
//		{
//			return m_it_ == rhs.m_it_;
//		}
//
//		bool operator!=(this_type const & rhs) const
//		{
//			return !(this->operator==(rhs));
//		}
//
//		this_type & operator ++()
//		{
//			++m_it_;
//			UpldateCellIteraor_();
//			return *this;
//		}
//		this_type operator ++(int)
//		{
//			this_type res(*this);
//			++res;
//			return std::move(res);
//		}
//	private:
//		void UpldateCellIteraor_()
//		{
//			c_it_ = base_container_type::find(m_it_);
//			while (c_it_ == base_container_type::end() && m_it_ != m_ie_)
//			{
//				++m_it_;
//				c_it_ = base_container_type::find(m_it_);
//			}
//		}
//	};
//
//	template<bool IsConst>
//	struct cell_range_: public mesh_type::range
//	{
//
//		typedef cell_iterator_<IsConst> iterator;
//		typedef cell_range_<IsConst> this_type;
//		typedef typename mesh_type::range mesh_range;
//		typedef typename std::conditional<IsConst, const container_type &, container_type&>::type container_reference;
//
//		container_reference data_;
//
//		cell_range_(container_reference data, mesh_range const & m_range) :
//				mesh_range(m_range), data_(data)
//		{
//		}
//
//		template<typename ...Args>
//		cell_range_(container_reference data,Args && ...args) :
//				mesh_range(std::forward<Args >(args)...), data_(data)
//		{
//		}
//
//		~cell_range_()
//		{
//		}
//
//		iterator begin() const
//		{
//			return iterator(data_, mesh_range::begin(), mesh_range::end());
//		}
//
//		iterator end() const
//		{
//
//			return iterator(data_, mesh_range::end(), mesh_range::end());
//		}
//
//		iterator rbegin() const
//		{
//			return iterator(data_, mesh_range::begin(), mesh_range::end());
//		}
//
//		iterator rend() const
//		{
//			return iterator(data_, mesh_range::rend(), mesh_range::end());
//		}
//
//		template<typename ...Args>
//		this_type split(Args const & ... args) const
//		{
//			return this_type(data_, mesh_range::split(std::forward<Args >(args)...));
//		}
//
//		template<typename ...Args>
//		this_type SubRange(Args const & ... args) const
//		{
//			return this_type(data_, mesh_range::SubRange(std::forward<Args >(args)...));
//		}
//		size_t size() const
//		{
//			size_t count = 0;
//			for (auto it = begin(), ie = end(); it != ie; ++it)
//			{
//				++count;
//			}
//			return count;
//		}
//	};
//	template<typename TV>
//	struct iterator_
//	{
//
//		typedef iterator_<TV> this_type;
//
///// One of the @link iterator_tags tag types@endlink.
//		typedef std::bidirectional_iterator_tag iterator_category;
//
///// The type "pointed to" by the iterator.
//		typedef particle_type value_type;
//
///// This type represents a pointer-to-value_type.
//		typedef value_type* pointer;
//
///// This type represents a reference-to-value_type.
//		typedef value_type& reference;
//
//		typedef cell_iterator_<value_type> cell_iterator;
//
//		typedef typename std::conditional<std::is_const<TV>::value, typename cell_iterator::value_type::const_iterator,
//		        typename cell_iterator::value_type::iterator>::type element_iterator;
//
//		cell_iterator c_it_;
//		element_iterator e_it_;
//
//		template<typename ...Args>
//		iterator_(Args const & ...args)
//				: c_it_(std::forward<Args >(args)...), e_it_(c_it_->begin())
//		{
//
//		}
//		iterator_(cell_iterator c_it, element_iterator e_it)
//				: c_it_(c_it), e_it_(e_it)
//		{
//
//		}
//		~iterator_()
//		{
//		}
//
//		reference operator*() const
//		{
//			return *e_it_;
//		}
//
//		pointer operator ->() const
//		{
//			return &(*e_it_);
//		}
//		bool operator==(this_type const & rhs) const
//		{
//			return c_it_ == rhs.c_it_ && (c_it_.isNull() || e_it_ == rhs.e_it_);
//		}
//
//		bool operator!=(this_type const & rhs) const
//		{
//			return !(this->operator==(rhs));
//		}
//
//		this_type & operator ++()
//		{
//			if (!c_it_.isNull())
//			{
//
//				if (e_it_ != c_it_->end())
//				{
//					++e_it_;
//
//				}
//				else
//				{
//					++c_it_;
//
//					if (!c_it_.isNull())
//						e_it_ = c_it_->begin();
//				}
//
//			}
//
//			return *this;
//		}
//		this_type operator ++(int)
//		{
//			this_type res(*this);
//			++res;
//			return std::move(res);
//		}
//		this_type & operator --()
//		{
//			if (!c_it_.isNull())
//			{
//
//				if (e_it_ != c_it_->rend())
//				{
//					--e_it_;
//
//				}
//				else
//				{
//					--c_it_;
//
//					if (!c_it_.isNull())
//						e_it_ = c_it_.rbegin();
//				}
//
//			}
//			return *this;
//		}
//		this_type operator --(int)
//		{
//			this_type res(*this);
//			--res;
//			return std::move(res);
//		}
//	};
//	template<typename TV>
//	struct range_: public cell_range_<TV>
//	{
//
//		typedef iterator_<TV> iterator;
//		typedef range_<TV> this_type;
//		typedef cell_range_<TV> cell_range_type;
//		typedef typename std::conditional<std::is_const<TV>::value, const container_type &, container_type&>::type container_reference;
//
//		range_(cell_range_type range)
//				: cell_range_type(range)
//		{
//		}
//		template<typename ...Args>
//		range_(container_reference d,Args && ... args)
//				: cell_range_type(d, std::forward<Args >(args)...)
//		{
//		}
//
//		~range_()
//		{
//		}
//
//		iterator begin() const
//		{
//			auto cit = cell_range_type::begin();
//			return iterator(cit, cit->begin());
//		}
//
//		iterator end() const
//		{
//			auto cit = cell_range_type::rbegin();
//			return iterator(cit, cit->end());
//		}
//
//		iterator rbegin() const
//		{
//			auto cit = cell_range_type::rbegin();
//			return iterator(cit, cit->rbegin());
//		}
//
//		iterator rend() const
//		{
//			auto cit = cell_range_type::begin();
//			return iterator(cit, cit->rend());
//		}
//
//		size_t size() const
//		{
//			size_t count = 0;
//			for (auto it = cell_range_type::begin(), ie = cell_range_type::end(); it != ie; ++it)
//			{
//				count += it->size();
//			}
//			return count;
//		}
//		template<typename ...Args>
//		this_type split(Args const & ... args) const
//		{
//			return this_type(cell_range_type::split(std::forward<Args >(args)...));
//		}
//	};
//	typedef iterator_<particle_type> iterator;
//	typedef iterator_<const particle_type> const_iterator;
//
//	typedef cell_iterator_<false> cell_iterator;
//	typedef cell_iterator_<true> const_cell_iterator;

//	typedef range_<particle_type> range;
//	typedef range_<const Point_s> const_range;

//	iterator begin()
//	{
//		return iterator(base_container_type::begin());
//	}
//
//	iterator end()
//	{
//		return iterator(base_container_type::rbegin(), base_container_type::rbegin()->end());
//	}
//
//	iterator rbegin()
//	{
//		return iterator(base_container_type::rbegin(), base_container_type::rbeing()->rbegin());
//	}
//
//	iterator rend()
//	{
//		return iterator(base_container_type::begin(), base_container_type::begin()->rend());
//	}
//
//	typename container_type::iterator cell_begin()
//	{
//		return (base_container_type::begin());
//	}
//
//	typename container_type::iterator cell_end()
//	{
//		return (base_container_type::end());
//	}
//	typename container_type::iterator cell_rbegin()
//	{
//		return (base_container_type::rbegin());
//	}
//
//	typename container_type::iterator cell_rend()
//	{
//		return (base_container_type::rend());
//	}
//	typename container_type::const_iterator cell_begin() const
//	{
//		return (base_container_type::cbegin());
//	}
//
//	typename container_type::const_iterator cell_end() const
//	{
//		return (base_container_type::cend());
//	}
//
//	typename container_type::const_iterator cell_rbegin() const
//	{
//		return (base_container_type::crbegin());
//	}
//
//	typename container_type::const_iterator cell_rend() const
//	{
//		return (base_container_type::crend());
//	}
#endif /* CORE_GTL_CONTAINER_SORTED_SET_H_ */
