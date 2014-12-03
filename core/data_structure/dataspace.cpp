/*
 * dataspace.cpp
 *
 *  Created on: 2014年11月13日
 *      Author: salmon
 */

#include "dataspace.h"
#include "data_type.h"
#include "../utilities/utilities.h"
#include "../parallel/distributed_array.h"

namespace simpla
{
struct DataSpace::pimpl_s
{
	pimpl_s();

	pimpl_s(pimpl_s const &);

	~pimpl_s();

	void swap(pimpl_s &);

	bool is_valid() const;

	Properties const &properties(std::string const& key = "") const;

	Properties &properties(std::string const& key = "");

	void init(size_t nd, size_t const * b, size_t const* e, size_t gw = 2);

	bool sync_ghosts(DataSet *ds, size_t flag);

	size_t num_of_dims() const;

	/**
	 * dimensions of global data
	 * @return <global start, global count>
	 */
	std::tuple<size_t const *, size_t const *> global_shape() const;

	/**
	 * dimensions of data in local memory
	 * @return <local start, local count>
	 */
	std::tuple<size_t const *, size_t const *> local_shape() const;

	/**
	 * logical shape of data in local memory, which  is the result of select_hyperslab
	 * @return <strat,count,strides,block>
	 */
	std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> shape() const;

	/**
	 *  select a hyper rectangle from local data
	 * @param start
	 * @param count
	 * @param strides
	 * @param block
	 * @return
	 */
	bool select_hyperslab(size_t const * start, size_t const * count,
			size_t const * strides = nullptr, size_t const * block = nullptr);
private:

	std::shared_ptr<DistributedArray> darray_;

	size_t * start_ = nullptr;
	size_t * count_ = nullptr;
	size_t * stride_ = nullptr;
	size_t * block_ = nullptr;

};
DataSpace::pimpl_s::pimpl_s()
{
}
DataSpace::pimpl_s::pimpl_s(pimpl_s const & other)
{
}
DataSpace::pimpl_s::~pimpl_s()
{
}
void DataSpace::pimpl_s::swap(pimpl_s &)
{
}

bool DataSpace::pimpl_s::is_valid() const
{
	return darray_ != nullptr && darray_->is_valid();
}
Properties & DataSpace::pimpl_s::properties(std::string const& key)
{
	return darray_->properties(key);
}
Properties const& DataSpace::pimpl_s::properties(std::string const& key) const
{
	return darray_->properties(key);
}

void DataSpace::pimpl_s::init(size_t nd, size_t const * b, size_t const* e,
		size_t gw)
{
	darray_ = std::make_shared<DistributedArray>(nd, b, e, gw);
}

bool DataSpace::pimpl_s::sync_ghosts(DataSet * ds, size_t flag)
{
	if (darray_ == nullptr)
		return false;

	return darray_->sync_ghosts(ds, flag);

}

size_t DataSpace::pimpl_s::num_of_dims() const
{
	return darray_->num_of_dims();
}

std::tuple<size_t const *, size_t const *> DataSpace::pimpl_s::global_shape() const
{
	return std::move(darray_->global_shape());
}
std::tuple<size_t const *, size_t const *> DataSpace::pimpl_s::local_shape() const
{
	return std::move(darray_->local_shape());
}

std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::pimpl_s::shape() const
{

	return std::forward_as_tuple(start_, count_, stride_, block_);
}
bool DataSpace::pimpl_s::select_hyperslab(size_t const * start,
		size_t const * count, size_t const * strides, size_t const * block)
{
	UNIMPLEMENTED;
	return true;
}
//===================================================================
DataSpace::DataSpace() :
		pimpl_(new pimpl_s)
{
}
DataSpace::DataSpace(DataSpace const & other) :
		pimpl_(new pimpl_s(*other.pimpl_))
{
}
DataSpace::~DataSpace()
{
	delete pimpl_;
}

void DataSpace::swap(DataSpace &other)
{
	pimpl_->swap(*other.pimpl_);
}

DataSpace DataSpace::create_simple(size_t rank, size_t const* count)
{
	DataSpace res;
	size_t start[rank];
	for (int i = 0; i < rank; ++i)
	{
		start[i] = 0;
	}
	res.init(rank, start, count);
	return std::move(res);
}

bool DataSpace::is_valid() const
{
	return pimpl_->is_valid();
}
Properties & DataSpace::properties(std::string const& key)
{
	return pimpl_->properties(key);
}
Properties const& DataSpace::properties(std::string const& key) const
{
	return pimpl_->properties(key);
}

void DataSpace::init(size_t nd, size_t const * b, size_t const* e, size_t gw)
{
	pimpl_->init(nd, b, e, gw);
}

bool DataSpace::sync_ghosts(DataSet *ds, size_t flag)
{
	return pimpl_->sync_ghosts(ds, flag);
}

size_t DataSpace::num_of_dims() const
{
	return pimpl_->num_of_dims();
}

std::tuple<size_t const *, size_t const *> DataSpace::global_shape() const
{
	return pimpl_->global_shape();
}

std::tuple<size_t const *, size_t const *> DataSpace::local_shape() const
{
	return pimpl_->local_shape();
}

std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::shape() const
{
	return pimpl_->shape();
}

bool DataSpace::select_hyperslab(size_t const * start, size_t const * count,
		size_t const * strides, size_t const * block)
{
	return pimpl_->select_hyperslab(start, count, strides, block);
}

}
// namespace simpla
