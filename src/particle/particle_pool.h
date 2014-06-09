/*
 * particle_pool.h
 *
 *  Created on: 2014年4月29日
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "../utilities/log.h"
#include "../utilities/type_utilites.h"

#include "../parallel/parallel.h"
#include "save_particle.h"

#ifndef NO_STD_CXX
//need  libstdc++
#include <ext/mt_allocator.h>
template<typename T> using FixedSmallSizeAlloc=__gnu_cxx::__mt_alloc<T>;
#endif

namespace simpla
{

//*******************************************************************************************************

template<typename TM, typename TParticle>
class ParticlePool
{
	std::mutex write_lock_;

public:
	static constexpr int IForm = VERTEX;

	typedef TM mesh_type;
	typedef TParticle particle_type;
	typedef ParticlePool<mesh_type, particle_type> this_type;
	typedef particle_type value_type;

	typedef typename mesh_type::iterator mesh_iterator;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	//container
//	template<typename TV> using Cell=std::list<TV, FixedSmallSizeAlloc<TV> >;

	typedef std::list<value_type, FixedSmallSizeAlloc<value_type> > cell_type;

	typedef std::map<mesh_iterator, cell_type> container_type;

	typedef typename cell_type::allocator_type allocator_type;

public:
	mesh_type const & mesh;
	//***************************************************************************************************
	// Constructor

	ParticlePool(mesh_type const & pmesh);

	template<typename ...Others> ParticlePool(mesh_type const & pmesh, Others const & ...);

	// Destructor
	~ParticlePool();

	template<typename TDict, typename ...Args> void Load(TDict const & dict, Args const & ...others);

	std::string Save(std::string const & path) const;

	container_type &data()
	{
		return data_;
	}
	container_type const&data() const
	{
		return data_;
	}

	//***************************************************************************************************

	allocator_type GetAllocator()
	{
		return allocator_;
	}

	cell_type GetCell()
	{
		return std::move(cell_type(allocator_));
	}

	cell_type & GetCell(container_type * c, mesh_iterator s)
	{
		auto it = c->find(s);

		if (it == c->end())
		{
			it = c->emplace(s, GetCell()).first;
		}

		return it->second;
	}

	inline void Insert(mesh_iterator s, particle_type p)
	{
		GetCell(&data_, s).emplace_back(p);
	}
	cell_type & operator[](mesh_iterator s)
	{
		return GetCell(&data_, s);
	}
	cell_type const & operator[](mesh_iterator s) const
	{
		return data_.at(s);
	}
	cell_type &at(mesh_iterator s)
	{
		return data_.at(s);
	}
	cell_type const & at(mesh_iterator s) const
	{
		return data_.at(s);
	}

private:

	template<bool IsConst>
	struct cell_iterator_
	{

		typedef cell_iterator_<IsConst> this_type;

		/// One of the @link iterator_tags tag types@endlink.
		typedef std::forward_iterator_tag iterator_category;

		/// The type "pointed to" by the iterator.
		typedef typename std::conditional<IsConst, const cell_type, cell_type>::type value_type;

		/// This type represents a pointer-to-value_type.
		typedef value_type * pointer;

		/// This type represents a reference-to-value_type.
		typedef value_type & reference;

		typedef typename std::conditional<IsConst, container_type const &, container_type&>::type container_reference;

		typedef typename std::conditional<IsConst, typename container_type::const_iterator,
		        typename container_type::iterator>::type cell_iterator;

		container_reference data_;
		mesh_iterator m_it_, m_ie_;
		cell_iterator c_it_;

		cell_iterator_(container_reference data, mesh_iterator m_ib, mesh_iterator m_ie)
				: data_(data), m_it_(m_ib), m_ie_(m_ie), c_it_(data_.find(m_ib))
		{
			UpldateCellIteraor_();
		}

		~cell_iterator_()
		{
		}

		reference operator*()
		{
			return c_it_->second;
		}
		const reference operator*() const
		{
			return c_it_->second;
		}
		pointer operator ->()
		{
			return &(c_it_->second);
		}
		const pointer operator ->() const
		{
			return &(c_it_->second);
		}
		bool operator==(this_type const & rhs) const
		{
			return m_it_ == rhs.m_it_;
		}

		bool operator!=(this_type const & rhs) const
		{
			return !(this->operator==(rhs));
		}

		this_type & operator ++()
		{
			++m_it_;
			UpldateCellIteraor_();
			return *this;
		}
		this_type operator ++(int)
		{
			this_type res(*this);
			++res;
			return std::move(res);
		}
	private:
		void UpldateCellIteraor_()
		{
			c_it_ = data_.find(m_it_);
			while (c_it_ == data_.end() && m_it_ != m_ie_)
			{
				++m_it_;
				c_it_ = data_.find(m_it_);
			}
			CHECK(m_it_ == m_ie_);
		}
	};

	template<bool IsConst>
	struct cell_range_: public mesh_type::range
	{

		typedef cell_iterator_<IsConst> iterator;
		typedef cell_range_<IsConst> this_type;
		typedef typename mesh_type::range mesh_range;
		typedef typename std::conditional<IsConst, const container_type &, container_type&>::type container_reference;

		container_reference data_;

		cell_range_(container_reference data, mesh_range const & m_range)
				: mesh_range(m_range), data_(data)
		{
		}

		template<typename ...Args>
		cell_range_(container_reference data, Args const & ...args)
				: mesh_range(std::forward<Args const &>(args)...), data_(data)
		{
		}

		~cell_range_()
		{
		}

		iterator begin() const
		{
			return iterator(data_, mesh_range::begin(), mesh_range::end());
		}

		iterator end() const
		{

			return iterator(data_, mesh_range::end(), mesh_range::end());
		}

		iterator rbegin() const
		{
			return iterator(data_, mesh_range::begin(), mesh_range::end());
		}

		iterator rend() const
		{
			return iterator(data_, mesh_range::rend(), mesh_range::end());
		}

		template<typename ...Args>
		this_type Split(Args const & ... args) const
		{
			return this_type(data_, mesh_range::Split(std::forward<Args const &>(args)...));
		}
		size_t size() const
		{
			size_t count = 0;
			for (auto it = begin(), ie = end(); it != ie; ++it)
			{
				++count;
			}
			return count;
		}
	};
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
//				: c_it_(std::forward<Args const &>(args)...), e_it_(c_it_->begin())
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
//		range_(container_reference d, Args const & ... args)
//				: cell_range_type(d, std::forward<Args const &>(args)...)
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
//		this_type Split(Args const & ... args) const
//		{
//			return this_type(cell_range_type::Split(std::forward<Args const &>(args)...));
//		}
//	};
public:

//	typedef iterator_<particle_type> iterator;
//	typedef iterator_<const particle_type> const_iterator;

	typedef cell_iterator_<false> cell_iterator;
	typedef cell_iterator_<true> const_cell_iterator;

//	typedef range_<particle_type> range;
//	typedef range_<const particle_type> const_range;

	typedef cell_range_<false> cell_range;
	typedef cell_range_<true> const_cell_range;

//	template<typename ... Args>
//	range GetRange(Args const & ... args)
//	{
//		return range(data_, mesh.GetRange(IForm, std::forward<Args const &>(args)...));
//	}
//
//	template<typename ... Args>
//	const_range GetRange(Args const & ... args) const
//	{
//		return const_range(data_, mesh.GetRange(IForm, std::forward<Args const &>(args)...));
//	}

	cell_range SelectCell()
	{
		return cell_range(data_, mesh.GetRange(IForm));
	}

	cell_range SelectCell() const
	{
		return cell_range(data_, mesh.GetRange(IForm));
	}
	template<typename ... Args>
	cell_range SelectCell(Args const & ... args)
	{
		return cell_range(data_, mesh.GetRange(IForm, std::forward<Args const &>(args)...));
	}

	template<typename ... Args>
	const_cell_range SelectCell(Args const & ... args) const
	{
		return const_cell_range(data_, mesh.GetRange(IForm, std::forward<Args const &>(args)...));
	}

//	iterator begin()
//	{
//		return iterator(data_.begin());
//	}
//
//	iterator end()
//	{
//		return iterator(data_.rbegin(), data_.rbegin()->end());
//	}
//
//	iterator rbegin()
//	{
//		return iterator(data_.rbegin(), data_.rbeing()->rbegin());
//	}
//
//	iterator rend()
//	{
//		return iterator(data_.begin(), data_.begin()->rend());
//	}

	typename container_type::iterator cell_begin()
	{
		return (data_.begin());
	}

	typename container_type::iterator cell_end()
	{
		return (data_.end());
	}
	typename container_type::iterator cell_rbegin()
	{
		return (data_.rbegin());
	}

	typename container_type::iterator cell_rend()
	{
		return (data_.rend());
	}
	typename container_type::const_iterator cell_begin() const
	{
		return (data_.cbegin());
	}

	typename container_type::const_iterator cell_end() const
	{
		return (data_.cend());
	}

	typename container_type::const_iterator cell_rbegin() const
	{
		return (data_.crbegin());
	}

	typename container_type::const_iterator cell_rend() const
	{
		return (data_.crend());
	}

	//***************************************************************************************************
	// Cell operation

	void Clear(cell_iterator it);

	void ClearEmpty();

	void Merge(container_type * other, container_type *dest = nullptr);

	void Add(cell_type *src);

	template<typename TRange>
	void Remove(TRange r, std::function<bool(particle_type const&)> const & filter, cell_type* other = nullptr);

	template<typename TRange>
	void Remove(TRange r, cell_type *other = nullptr);

	void Remove(cell_iterator it, cell_type *other = nullptr);

//***************************************************************************************************

	void Sort();

	bool is_sorted() const
	{
		return isSorted_;
	}

	void EnableSort()
	{
		isSorted_ = false;
	}

	size_t size() const
	{
		size_t res = 0;

		for (auto const & v : data_)
		{
			res += v.second.size();
		}
		return res;
	}

	container_type const & GetTree() const
	{
		return data_;
	}

	void WriteLock()
	{
		write_lock_.lock();
	}
	void WriteUnLock()
	{
		write_lock_.unlock();
	}

	bool isSorted_;

	allocator_type allocator_;
	container_type data_;

	/**
	 *  resort particles in cell 's', and move out boundary particles to 'dest' container
	 * @param
	 */
	template<typename TSrc, typename TDest> void Sort_(TSrc *, TDest *dest);

};

/***
 * FIXME (salmon):  We need a  thread-safe and  high performance allocator for std::map<mesh_iterator,std::list<allocator> > !!
 */
template<typename TM, typename TParticle>
ParticlePool<TM, TParticle>::ParticlePool(mesh_type const & pmesh)
		: mesh(pmesh), isSorted_(false), allocator_()
{

}
template<typename TM, typename TParticle>
template<typename ...Others>
ParticlePool<TM, TParticle>::ParticlePool(mesh_type const & pmesh, Others const & ...others)
		: ParticlePool(pmesh)
{
	Load(std::forward<Others const &>(others)...);
}
template<typename TM, typename TParticle>
template<typename TDict, typename ...Args> void ParticlePool<TM, TParticle>::Load(TDict const & dict,
        Args const & ...others)
{

}

template<typename TM, typename TParticle>
ParticlePool<TM, TParticle>::~ParticlePool()
{
}

template<typename TM, typename TParticle>
std::string ParticlePool<TM, TParticle>::Save(std::string const & name) const
{
	return simpla::Save(name, *this);
}

template<typename TM, typename TParticle>
template<typename TSrc, typename TDest>
void ParticlePool<TM, TParticle>::Sort_(TSrc * p_src, TDest *p_dest_contianer)
{

	auto pt = p_src->begin();

	auto shift = mesh.GetShift(IForm);

	while (pt != p_src->end())
	{
		auto p = pt;
		++pt;

		mesh_iterator id_dest = mesh.CoordinatesGlobalToLocalDual(&(p->x), shift);

		p->x = mesh.CoordinatesLocalToGlobal(id_dest, p->x);
		auto & dest = GetCell(p_dest_contianer, id_dest);
		dest.splice(dest.begin(), *p_src, p);

	}

}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Sort()
{

	if (is_sorted())
		return;

//	ParallelDo(
//
//	[this](int t_num,int t_id)
//	{
//		container_type dest;
//		for (auto s : mesh.GetRange(IForm).Split(t_num,t_id))
//		{
//
// 			CHECK(mesh.Hash(s));
//			auto it = data_.find(s);
//
//			if (it != data_.end()) this->Sort_(&(it->second), &dest);
//		}
//		Merge(&dest);
//	}
//
//	);

	//FIXME Here should be PARALLEL (multi-thread)
	container_type dest;
	for (auto s : mesh.GetRange(IForm))
	{

		auto it = data_.find(s);

		if (it != data_.end())
			this->Sort_(&(it->second), &dest);
	}
	Merge(&dest, &data_);
	isSorted_ = true;

}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::ClearEmpty()
{
	write_lock_.lock();
	auto it = data_.begin(), ie = data_.end();

	while (it != ie)
	{
		auto t = it;
		++it;
		if (t->second.empty())
		{
			data_.erase(t);
		}
	}
	write_lock_.unlock();
}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Clear(cell_iterator it)
{
	write_lock_.lock();
	data_.erase(it.c_it_);
	write_lock_.unlock();
}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Merge(container_type * other, container_type *dest)
{
	if (dest == nullptr)
		dest = &data_;

	write_lock_.lock();
	for (auto & v : *other)
	{
		auto & c = GetCell(dest, v.first);
		c.splice(c.begin(), v.second);
	}
	write_lock_.unlock();

}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Add(cell_type* other)
{
	Sort_(other, &data_);
}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Remove(cell_iterator it, cell_type *other)
{
	if (other == nullptr)
	{
		it->clear();
	}
	else
	{
		other->splice(other->begin(), *it);
	}
}
template<typename TM, typename TParticle>
template<typename TRange>
void ParticlePool<TM, TParticle>::Remove(TRange r, cell_type * other)
{
	for (auto it = r.begin(), ie = r.end(); it != ie; ++it)
	{
		Remove(it, other);
	}
}

template<typename TM, typename TParticle>
template<typename TRange>
void ParticlePool<TM, TParticle>::Remove(TRange r, std::function<bool(particle_type const&)> const & filter,
        cell_type * other)
{

	for (auto & cell : r)
	{
		auto it = cell.begin(), ie = cell.end();

		while (it != ie)
		{
			auto p = it;
			++it;
			if (filter(*p))
			{
				if (other == nullptr)
				{
					cell.erase(it);
				}
				else
				{
					other->splice(other->begin(), cell, p);
				}
			}

		}
	}
}

}  // namespace simpla

#endif /* PARTICLE_POOL_H_ */
