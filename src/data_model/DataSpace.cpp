/**
 * @file DataSpace.cpp
 *
 *  Created on: 2014-11-13
 *      @author: salmon
 */

#include <algorithm>
#include <utility>
#include <tuple>

#include "../gtl/ntuple.h"
#include "../gtl/utilities/utilities.h"
#include "../parallel/Parallel.h"
#include "DataSpace.h"

namespace simpla { namespace data_model
{
struct DataSpace::pimpl_s
{

    /**
     *
     *   a----------------------------b
     *   |                            |
     *   |     c--------------d       |
     *   |     |              |       |
     *   |     |  e*******f   |       |
     *   |     |  *       *   |       |
     *   |     |  *       *   |       |
     *   |     |  *       *   |       |
     *   |     |  *********   |       |
     *   |     ----------------       |
     *   ------------------------------
     *
     *   a=0
     *   b-a = dimension
     *   e-a = offset
     *   f-e = count
     *   d-c = local_dimension
     *   c-a = local_offset
     */

    data_shape_s m_d_shape_;
    std::vector<size_t> m_selected_points_;
    // index_tuple m_local_dimensions_;
    // index_tuple m_local_offset_;

};

//===================================================================

DataSpace::DataSpace() : m_pimpl_{new pimpl_s} { }

DataSpace::DataSpace(int ndims, index_type const *dims) :
        m_pimpl_(new pimpl_s)
{

    std::get<0>(m_pimpl_->m_d_shape_)/*ndims      */ = ndims;
    std::get<1>(m_pimpl_->m_d_shape_)/*dimensions */ = dims;
    std::get<2>(m_pimpl_->m_d_shape_)/*start     */  = 0;
    std::get<3>(m_pimpl_->m_d_shape_)/*stride     */ = 1;
    std::get<4>(m_pimpl_->m_d_shape_)/*count      */ = dims;
    std::get<5>(m_pimpl_->m_d_shape_)/*block      */ = 1;


}

DataSpace::DataSpace(const DataSpace &other) :
        m_pimpl_(new pimpl_s)
{
    // m_pimpl_->m_d_shape_.ndims = other.m_pimpl_->m_d_shape_.ndims;
    // m_pimpl_->m_d_shape_.dimensions = other.m_pimpl_->m_d_shape_.dimensions;
    // m_pimpl_->m_d_shape_.offset = other.m_pimpl_->m_d_shape_.offset;
    // m_pimpl_->m_d_shape_.count = other.m_pimpl_->m_d_shape_.count;
    // m_pimpl_->m_d_shape_.stride = other.m_pimpl_->m_d_shape_.stride;
    // m_pimpl_->m_d_shape_.block = other.m_pimpl_->m_d_shape_.block;

    m_pimpl_->m_d_shape_ = other.m_pimpl_->m_d_shape_;

//	m_pimpl_->m_local_dimensions_ = other.m_pimpl_->m_local_dimensions_;
//	m_pimpl_->m_local_offset_ = other.m_pimpl_->m_local_offset_;


}

DataSpace::DataSpace(DataSpace &&other) : m_pimpl_(other.m_pimpl_) { }


DataSpace::~DataSpace()
{
}

void DataSpace::swap(DataSpace &other)
{
    std::swap(m_pimpl_, other.m_pimpl_);
}

DataSpace DataSpace::create_simple(int ndims, const index_type *dims)
{
    return std::move(DataSpace(ndims, dims));
}

std::tuple<DataSpace, DataSpace> DataSpace::create_simple_unordered(size_t count)
{
    size_t offset = 0;
    size_t total_count = count;
    std::tie(offset, total_count) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(count));
    DataSpace memory_space = data_model::DataSpace::create_simple(1, &count);

    DataSpace data_space = data_model::DataSpace::create_simple(1, &total_count);
    data_space.select_hyperslab(&offset, nullptr, &count, nullptr);

    return std::forward_as_tuple(data_space, memory_space);
}

//
//std::tuple<DataSpace, DataSpace>  DataSpace::create(
//        size_t rank,
//        index_type const *dims,
//        index_type const *start,
//        index_type const *_stride,
//        index_type const *count,
//        index_type const *_block)
//{
//
//    DataSpace data_space, memory_space;
//
//    if (dims == nullptr && start == nullptr)
//    {
//        size_t count = rank;
//        size_t offset = 0;
//        size_t total_count = count;
//
//        std::tie(offset, total_count) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(count));
//
//        data_space = DataSpace::create_simple(1, &total_count);
//        data_space.select_hyperslab(&offset, nullptr, &count, nullptr);
//        memory_space = DataSpace::create_simple(1, &count);
//
//    }
//    else
//    {
//        if (dims != nullptr)
//        {
//            data_space = DataSpace::create_simple(static_cast<int>(rank), dims);
//            memory_space = data_space;
//        }
//        else
//        {   //fixme calculate distributed array dimensions
//            UNIMPLEMENTED2("fixme calculate distributed array dimensions");
//        }
//    }
//
//    return std::forward_as_tuple(data_space, memory_space);
//}


bool DataSpace::is_valid() const
{
    return (!!(m_pimpl_))
           &&
           (std::get<2>(m_pimpl_->m_d_shape_) + std::get<4>(m_pimpl_->m_d_shape_) <= std::get<1>(m_pimpl_->m_d_shape_));
}

bool DataSpace::is_simple() const
{
    return m_pimpl_->m_selected_points_.size() == 0;
}

bool DataSpace::is_full() const { return size() == num_of_elements(); }

DataSpace::data_shape_s const &DataSpace::shape() const
{
    return m_pimpl_->m_d_shape_;
}

DataSpace::data_shape_s &DataSpace::shape()
{
    return m_pimpl_->m_d_shape_;
}

size_t DataSpace::size() const
{
    size_t s = 1;

    int ndims = std::get<0>(m_pimpl_->m_d_shape_);

    auto const &dims = std::get<1>(m_pimpl_->m_d_shape_);

    for (int i = 0; i < ndims; ++i)
    {
        s *= dims[i];
    }
    return s;
}

size_t DataSpace::num_of_elements() const
{

    if (!is_simple())
    {
        return m_pimpl_->m_selected_points_.size() / std::get<0>(m_pimpl_->m_d_shape_);
    }
    else
    {
        size_t s = 1;

        int ndims = std::get<0>(m_pimpl_->m_d_shape_);

        auto const &count = std::get<4>(m_pimpl_->m_d_shape_);

        for (int i = 0; i < ndims; ++i) { s *= count[i]; }

        return s;
    }

}

std::vector<size_t> const &DataSpace::selected_points() const { return m_pimpl_->m_selected_points_; };

std::vector<size_t> &DataSpace::selected_points() { return m_pimpl_->m_selected_points_; };

void DataSpace::select_point(const size_t *idx)
{
    int ndims = std::get<0>(m_pimpl_->m_d_shape_);

    for (int i = 0; i < ndims; ++i) { m_pimpl_->m_selected_points_.push_back(idx[i]); }

}

void DataSpace::select_point(size_t pos) { m_pimpl_->m_selected_points_.push_back(pos); }


void DataSpace::select_points(size_t num, const size_t *tags)
{
    int ndims = std::get<0>(this->shape());

    size_t head = m_pimpl_->m_selected_points_.size();
    size_t tail = head + num;
    m_pimpl_->m_selected_points_.resize(tail);
    parallel::parallel_for(
            parallel::blocked_range<size_t>(head, tail),
            [&](parallel::blocked_range<size_t> const &r)
            {
                for (size_t i = r.begin(), ie = r.end(); i != ie; ++i)
                {
                    for (size_t j = 0; j < ndims; ++j)
                    {
                        m_pimpl_->m_selected_points_[i * ndims + j] = tags[(i - head) * ndims + j];
                    }
                }
            }

    );
}

DataSpace &DataSpace::select_hyperslab(index_type const *start,
                                       index_type const *_stride,
                                       index_type const *count,
                                       index_type const *_block)
{
    if (!is_valid()) { RUNTIME_ERROR << ("data_space is invalid!"); }

    if (start != nullptr) { std::get<2>(m_pimpl_->m_d_shape_) = start; }

    if (_stride != nullptr) { std::get<3>(m_pimpl_->m_d_shape_) *= _stride; }

    if (count != nullptr) { std::get<4>(m_pimpl_->m_d_shape_) = count; }

    if (_block != nullptr) { std::get<5>(m_pimpl_->m_d_shape_) *= _block; }

    return *this;

}

void DataSpace::clear_selected()
{
    m_pimpl_->m_selected_points_.clear();

    std::get<2>(m_pimpl_->m_d_shape_) = 0;//start;

    std::get<3>(m_pimpl_->m_d_shape_) = 1;//_stride;

    std::get<4>(m_pimpl_->m_d_shape_) = std::get<1>(m_pimpl_->m_d_shape_);//count;

    std::get<5>(m_pimpl_->m_d_shape_) = 1;// _block;
}

std::ostream &DataSpace::print(std::ostream &os, int indent) const
{
    return base::Object::print(os, indent);
}
//bool data_space::is_distributed() const
//{
//	bool flag = false;
//	for (int i = 0; i < m_pimpl_->m_d_shape_.ndims; ++i)
//	{
//		if (m_pimpl_->m_d_shape_.dimensions[i] != m_pimpl_->m_local_dimensions_[i])
//		{
//			flag = true;
//			break;
//		};
//	}
//	return flag;
//}



//data_space::data_shape_s data_space::local_shape() const
//{
//
//	data_shape_s res = m_pimpl_->m_d_shape_;
//
//	res.dimensions = m_pimpl_->m_local_dimensions_;
//
//	res.offset = m_pimpl_->m_d_shape_.offset - m_pimpl_->m_local_offset_;
//
//	return std::move(res);
//}
//
//data_space::data_shape_s data_space::global_shape() const
//{
//	return m_pimpl_->m_d_shape_;
//}
//
//
//size_t data_space::local_memory_size() const
//{
//	size_t s = 1;
//
//	for (int i = 0; i < m_pimpl_->m_d_shape_.ndims; ++i)
//	{
//		s *= m_pimpl_->m_local_dimensions_[i];
//	}
//	return s;
//}
//
//data_space &data_space::set_local_shape(index_type const *local_dimensions =
//nullptr, index_type const *local_offset = nullptr)
//{
//
//	if (local_offset != nullptr)
//	{
//		m_pimpl_->m_local_offset_ = local_offset;
//
//	}
//	else
//	{
//		m_pimpl_->m_local_offset_ = m_pimpl_->m_d_shape_.offset;
//
//	}
//
//	if (local_dimensions != nullptr)
//	{
//		m_pimpl_->m_local_dimensions_ = local_dimensions;
//	}
//	else
//	{
//		m_pimpl_->m_local_dimensions_ = m_pimpl_->m_d_shape_.dimensions;
//	}
//
//	return *this;
//}
//
//void data_space::decompose(size_t ndims, size_t const * proc_dims,
//		size_t const * proc_coord)
//{
//	if (!is_valid())
//	{
//		THROW_EXCEPTION_RUNTIME_ERROR("data_space is invalid!");
//	}
//	if (ndims > m_pimpl_->m_ndims_)
//	{
//		THROW_EXCEPTION_RUNTIME_ERROR("data_space is too small to decompose!");
//	}
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> offset, count;
//	offset = 0;
//	count = m_pimpl_->m_count_;
//
//	for (int n = 0; n < ndims; ++n)
//	{
//
//		offset[n] = m_pimpl_->m_count_[n] * proc_coord[n] / proc_dims[n];
//		count[n] = m_pimpl_->m_count_[n] * (proc_coord[n] + 1) / proc_dims[n]
//				- offset[n];
//
//		if (count[n] <= 0)
//		{
//			THROW_EXCEPTION_RUNTIME_ERROR(
//					"data_space decompose fail! Dimension  is smaller than process grid. "
//							"[dimensions= "
//							+ value_to_string(m_pimpl_->m_dimensions_)
//							+ ", process dimensions="
//							+ value_to_string(proc_dims));
//		}
//	}
//
//	select_hyperslab(&offset[0], nullptr, &count[0], nullptr);
//
//	m_pimpl_->m_dimensions_ = (m_pimpl_->m_count_ + m_pimpl_->m_ghost_width_ * 2)
//			* m_pimpl_->m_stride_;
//	m_pimpl_->m_offset_ = m_pimpl_->m_ghost_width_ * m_pimpl_->m_stride_;
//}
//
//void decomposer_(size_t num_process, size_t process_num, size_t gw,
//		size_t ndims, size_t const *global_start, size_t const * global_count,
//		size_t * local_outer_start, size_t * local_outer_count,
//		size_t * local_inner_start, size_t * local_inner_count)
//{
//
//	for (int i = 0; i < ndims; ++i)
//	{
//		local_outer_count[i] = global_count[i];
//		local_outer_start[i] = global_start[i];
//		local_inner_count[i] = global_count[i];
//		local_inner_start[i] = global_start[i];
//	}
//
//	if (num_process <= 1)
//		return;
//
//	int n = 0;
//	index_type L = 0;
//	for (int i = 0; i < ndims; ++i)
//	{
//		if (global_count[i] > L)
//		{
//			L = global_count[i];
//			n = i;
//		}
//	}
//
//	if ((2 * gw * num_process > global_count[n] || num_process > global_count[n]))
//	{
//
//		THROW_EXCEPTION_RUNTIME_ERROR("Array is too small to split");
//
////		if (process_num > 0)
////			local_outer_end = local_outer_begin;
//	}
//	else
//	{
//		local_inner_start[n] = (global_count[n] * process_num) / num_process
//				+ global_start[n];
//		local_inner_count[n] = (global_count[n] * (process_num + 1))
//				/ num_process + global_start[n];
//		local_outer_start[n] = local_inner_start[n] - gw;
//		local_outer_count[n] = local_inner_count[n] + gw;
//	}
//
//}
//
//void data_space::pimpl_s::decompose()
//{
//
////	local_shape_.dimensions = global_shape_.dimensions;
////	local_shape_.count = global_shape_.count;
////	local_shape_.offset = global_shape_.offset;
////	local_shape_.stride = global_shape_.stride;
////	local_shape_.block = global_shape_.block;
//
////	if (!GLOBAL_COMM.is_valid()) return;
////
////	int num_process = GLOBAL_COMM.get_size();
////	unsigned int process_num = GLOBAL_COMM.get_rank();
////
////	decomposer_(num_process, process_num, gw_, ndims_,  //
////			&global_shape_.offset[0], &global_shape_.count[0],  //
////			&local_outer_shape_.offset[0], &local_outer_shape_.count[0],  //
////			&local_inner_shape_.offset[0], &local_inner_shape_.count[0]);
////
////	self_id_ = (process_num);
////
////	for (int dest = 0; dest < num_process; ++dest)
////	{
////		if (dest == self_id_)
////			continue;
////
////		sub_array_s node;
////
////		decomposer_(num_process, dest, gw, ndims_, &global_shape_.offset[0],
////				&global_shape_.count[0], &node.outer_offset[0],
////				&node.outer_count[0], &node.inner_offset[0],
////				&node.inner_count[0]
////
////				);
////
////		sub_array_s remote;
////
////		for (unsigned index_type s = 0, s_e = (1UL << (ndims_ * 2)); s < s_e; ++s)
////		{
////			remote = node;
////
////			bool is_duplicate = false;
////
////			for (int i = 0; i < ndims_; ++i)
////			{
////
////				int n = (s >> (i * 2)) & 3UL;
////
////				if (n == 3)
////				{
////					is_duplicate = true;
////					continue;
////				}
////
////				auto L = global_shape_.count[i] * ((n + 1) % 3 - 1);
////
////				remote.outer_offset[i] += L;
////				remote.inner_offset[i] += L;
////
////			}
////			if (!is_duplicate)
////			{
////				bool f_inner = Clipping(ndims_, local_outer_shape_.offset,
////						local_outer_shape_.count, remote.inner_offset,
////						remote.inner_count);
////				bool f_outer = Clipping(ndims_, local_inner_shape_.offset,
////						local_inner_shape_.count, remote.outer_offset,
////						remote.outer_count);
////
////				bool flag = f_inner && f_outer;
////
////				for (int i = 0; i < ndims_; ++i)
////				{
////					flag = flag && (remote.outer_count[i] != 0);
////				}
////				if (flag)
////				{
////					send_recv_.emplace_back(
////							send_recv_s(
////									{ dest, hash(&remote.outer_offset[0]), hash(
////											&remote.inner_offset[0]),
////											remote.outer_offset,
////											remote.outer_count,
////											remote.inner_offset,
////											remote.inner_count }));
////				}
////			}
////		}
////	}
//
//	is_valid_ = true;
//}

//bool data_space::sync(std::shared_ptr<void> data, DataType const & DataType,
//		size_t flag)
//{
//#if  !NO_MPI || USE_MPI
//	if (!GLOBAL_COMM.is_valid() || m_pimpl_->send_recv_.size() == 0)
//	{
//		return true;
//	}
//
//	MPI_Comm comm = GLOBAL_COMM.comm();
//
//	MPI_Request request[m_pimpl_->send_recv_.size() * 2];
//
//	int count = 0;
//
//	for (auto const & item : m_pimpl_->send_recv_)
//	{
//
//		MPIDataType send_type = MPIDataType::create(DataType, m_pimpl_->local_shape_.ndims ,
//		&m_pimpl_->local_shape_.dimensions[0], & item.send.offset[0],
//		&item.send.stride[0], &item.send.count[0], &item.send.block[0]);
//
//		dims_type recv_offset;
//		recv_offset = item.recv.offset - m_pimpl_->local_shape_.offset;
//
//		MPIDataType recv_type = MPIDataType::create(DataType, m_pimpl_->local_shape_.ndims ,
//		&m_pimpl_->local_shape_.dimensions[0], & item.recv.offset[0],
//		&item.recv.stride[0], &item.recv.count[0], &item.recv.block[0]);
//
//		MPI_Isend(data.get(), 1, send_type.type(), item.dest, item.send_tag,
//		comm, &request[count * 2]);
//		MPI_Irecv(data.get(), 1, recv_type.type(), item.dest, item.recv_tag,
//		comm, &request[count * 2 + 1]);
//
//		++count;
//	}
//
//	MPI_Waitall(m_pimpl_->send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);
//
//#endif //#if  !NO_MPI || USE_MPI
//
//	return true;
//}


}}//namespace simpla { namespace data_model

