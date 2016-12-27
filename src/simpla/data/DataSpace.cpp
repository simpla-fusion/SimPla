/**
 * @file DataSpace.cpp
 *
 *  Created on: 2014-11-13
 *      @author: salmon
 */

#include <algorithm>
#include <utility>
#include <tuple>

#include <simpla/algebra/nTuple.h>

#include "DataSpace.h"

namespace simpla { namespace data
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
    std::vector<size_type> m_selected_points_;
    // index_tuple m_local_dimensions_;
    // index_tuple m_local_offset_;

};

//===================================================================

DataSpace::DataSpace()
        : m_pimpl_{new pimpl_s} {}

DataSpace::DataSpace(int ndims, size_type const *dims)
        : m_pimpl_(new pimpl_s)
{

    std::get<0>(m_pimpl_->m_d_shape_)/*m_ndims_      */ = ndims;
    std::get<1>(m_pimpl_->m_d_shape_)/*dimensions */ = dims;
    std::get<2>(m_pimpl_->m_d_shape_)/*start     */  = 0;
    std::get<3>(m_pimpl_->m_d_shape_)/*stride     */ = 1;
    std::get<4>(m_pimpl_->m_d_shape_)/*count      */ = dims;
    std::get<5>(m_pimpl_->m_d_shape_)/*block      */ = 1;

}

DataSpace::DataSpace(const DataSpace &other)
        :
        m_pimpl_(new pimpl_s)
{
    // m_self_->m_d_shape_.m_ndims_ = other.m_self_->m_d_shape_.m_ndims_;
    // m_self_->m_d_shape_.dimensions = other.m_self_->m_d_shape_.dimensions;
    // m_self_->m_d_shape_.m_global_start_ = other.m_self_->m_d_shape_.m_global_start_;
    // m_self_->m_d_shape_.count = other.m_self_->m_d_shape_.count;
    // m_self_->m_d_shape_.stride = other.m_self_->m_d_shape_.stride;
    // m_self_->m_d_shape_.block = other.m_self_->m_d_shape_.block;

    m_pimpl_->m_d_shape_ = other.m_pimpl_->m_d_shape_;

//	m_self_->m_local_dimensions_ = other.m_self_->m_local_dimensions_;
//	m_self_->m_local_offset_ = other.m_self_->m_local_offset_;


}

DataSpace::DataSpace(DataSpace &&other)
        : m_pimpl_(other.m_pimpl_) {}

DataSpace::~DataSpace()
{
}

void DataSpace::swap(DataSpace &other)
{
    std::swap(m_pimpl_, other.m_pimpl_);
}

DataSpace DataSpace::create_simple(int ndims, const size_type *dims)
{
    return std::move(DataSpace(ndims, dims));
}

std::tuple<DataSpace, DataSpace> DataSpace::create_simple_unordered(size_type count)
{
    size_type offset = 0;
    size_type total_count = count;
//    std::tie(offset, total_count) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(count));
    DataSpace memory_space = DataSpace::create_simple(1, &count);

    DataSpace data_space = DataSpace::create_simple(1, &total_count);
    data_space.select_hyperslab(&offset, nullptr, &count, nullptr);

    return std::forward_as_tuple(data_space, memory_space);
}

//
//std::tuple<DataSpace, DataSpace>  DataSpace::clone(
//        size_type rank,
//        size_type const *topology_dims,
//        size_type const *start,
//        size_type const *_stride,
//        size_type const *count,
//        size_type const *_block)
//{
//
//    DataSpace data_space, memory_space;
//
//    if (topology_dims == nullptr && start == nullptr)
//    {
//        size_type count = rank;
//        size_type m_global_start_ = 0;
//        size_type total_count = count;
//
//        std::tie(m_global_start_, total_count) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(count));
//
//        data_space = DataSpace::create_simple(1, &total_count);
//        data_space.select_hyperslab(&m_global_start_, nullptr, &count, nullptr);
//        memory_space = DataSpace::create_simple(1, &count);
//
//    }
//    else
//    {
//        if (topology_dims != nullptr)
//        {
//            data_space = DataSpace::create_simple(static_cast<int>(rank), topology_dims);
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
    bool res = (m_pimpl_ != nullptr);
    if (res)
    {
        int ndims = std::get<0>(m_pimpl_->m_d_shape_);
        index_tuple t_dims = std::get<1>(m_pimpl_->m_d_shape_);
        index_tuple t_start = std::get<2>(m_pimpl_->m_d_shape_);
        index_tuple t_count = std::get<4>(m_pimpl_->m_d_shape_);

        for (int i = 0; i < ndims; ++i)
        {
            if (t_start[i] + t_count[i] > t_dims[i])
            {
                res = false;
                break;
            };
        }
    }


    return res;
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

size_type DataSpace::size() const
{
    size_type s = 1;

    int ndims = std::get<0>(m_pimpl_->m_d_shape_);

    auto const &dims = std::get<1>(m_pimpl_->m_d_shape_);

    for (int i = 0; i < ndims; ++i)
    {
        s *= dims[i];
    }
    return s;
}

size_type DataSpace::num_of_elements() const
{

    if (!is_simple())
    {
        return m_pimpl_->m_selected_points_.size() / std::get<0>(m_pimpl_->m_d_shape_);
    } else
    {
        size_type s = 1;

        int ndims = std::get<0>(m_pimpl_->m_d_shape_);

        auto const &count = std::get<4>(m_pimpl_->m_d_shape_);

        for (int i = 0; i < ndims; ++i) { s *= count[i]; }

        return s;
    }

}

std::vector<size_type> const &DataSpace::selected_points() const { return m_pimpl_->m_selected_points_; };

std::vector<size_type> &DataSpace::selected_points() { return m_pimpl_->m_selected_points_; };

void DataSpace::select_point(const size_type *idx)
{
    int ndims = std::get<0>(m_pimpl_->m_d_shape_);

    for (int i = 0; i < ndims; ++i) { m_pimpl_->m_selected_points_.push_back(idx[i]); }

}

void DataSpace::select_point(size_type pos) { m_pimpl_->m_selected_points_.push_back(pos); }

void DataSpace::select_points(size_type num, const size_type *tags)
{
    int ndims = std::get<0>(this->shape());

    size_type head = m_pimpl_->m_selected_points_.size();
    size_type tail = head + num;
    m_pimpl_->m_selected_points_.resize(tail);
    parallel::parallel_for(
            parallel::blocked_range<size_type>(head, tail),
            [&](parallel::blocked_range <size_type> const &r)
            {
                for (size_type i = r.begin(), ie = r.end(); i != ie; ++i)
                {
                    for (size_type j = 0; j < ndims; ++j)
                    {
                        m_pimpl_->m_selected_points_[i * ndims + j] = tags[(i - head) * ndims + j];
                    }
                }
            }

    );
}

DataSpace &DataSpace::select_hyperslab(size_type const *start,
                                       size_type const *_stride,
                                       size_type const *count,
                                       size_type const *_block)
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
    return os;
}
//bool data_space::is_distributed() const
//{
//	bool id = false;
//	for (int i = 0; i < m_self_->m_d_shape_.m_ndims_; ++i)
//	{
//		if (m_self_->m_d_shape_.dimensions[i] != m_self_->m_local_dimensions_[i])
//		{
//			id = true;
//			break;
//		};
//	}
//	return id;
//}



//data_space::data_shape_s data_space::local_shape() const
//{
//
//	data_shape_s res = m_self_->m_d_shape_;
//
//	res.dimensions = m_self_->m_local_dimensions_;
//
//	res.m_global_start_ = m_self_->m_d_shape_.m_global_start_ - m_self_->m_local_offset_;
//
//	return std::move(res);
//}
//
//data_space::data_shape_s data_space::global_shape() const
//{
//	return m_self_->m_d_shape_;
//}
//
//
//size_type data_space::local_memory_size() const
//{
//	size_type s = 1;
//
//	for (int i = 0; i < m_self_->m_d_shape_.m_ndims_; ++i)
//	{
//		s *= m_self_->m_local_dimensions_[i];
//	}
//	return s;
//}
//
//data_space &data_space::set_local_shape(size_type const *local_dimensions =
//nullptr, size_type const *local_offset = nullptr)
//{
//
//	if (local_offset != nullptr)
//	{
//		m_self_->m_local_offset_ = local_offset;
//
//	}
//	else
//	{
//		m_self_->m_local_offset_ = m_self_->m_d_shape_.m_global_start_;
//
//	}
//
//	if (local_dimensions != nullptr)
//	{
//		m_self_->m_local_dimensions_ = local_dimensions;
//	}
//	else
//	{
//		m_self_->m_local_dimensions_ = m_self_->m_d_shape_.dimensions;
//	}
//
//	return *this;
//}
//
//void data_space::decompose(size_type m_ndims_, size_type const * proc_dims,
//		size_type const * proc_coord)
//{
//	if (!is_valid())
//	{
//		THROW_EXCEPTION_RUNTIME_ERROR("data_space is invalid!");
//	}
//	if (m_ndims_ > m_self_->m_topology_ndims_)
//	{
//		THROW_EXCEPTION_RUNTIME_ERROR("data_space is too small to decompose!");
//	}
//	nTuple<size_type, MAX_NDIMS_OF_ARRAY> m_global_start_, count;
//	m_global_start_ = 0;
//	count = m_self_->m_count_;
//
//	for (int n = 0; n < m_ndims_; ++n)
//	{
//
//		m_global_start_[n] = m_self_->m_count_[n] * proc_coord[n] / proc_dims[n];
//		count[n] = m_self_->m_count_[n] * (proc_coord[n] + 1) / proc_dims[n]
//				- m_global_start_[n];
//
//		if (count[n] <= 0)
//		{
//			THROW_EXCEPTION_RUNTIME_ERROR(
//					"data_space decompose fail! Dimension  is smaller than process grid. "
//							"[dimensions= "
//							+ value_to_string(m_self_->m_dimensions_)
//							+ ", process dimensions="
//							+ value_to_string(proc_dims));
//		}
//	}
//
//	select_hyperslab(&m_global_start_[0], nullptr, &count[0], nullptr);
//
//	m_self_->m_dimensions_ = (m_self_->m_count_ + m_self_->m_ghost_width_ * 2)
//			* m_self_->m_stride_;
//	m_self_->m_offset_ = m_self_->m_ghost_width_ * m_self_->m_stride_;
//}
//
//void decomposer_(size_type num_process, size_type process_num, size_type gw,
//		size_type m_ndims_, size_type const *m_global_start_, size_type const * global_count,
//		size_type * local_outer_start, size_type * local_outer_count,
//		size_type * local_inner_start, size_type * local_inner_count)
//{
//
//	for (int i = 0; i < m_ndims_; ++i)
//	{
//		local_outer_count[i] = global_count[i];
//		local_outer_start[i] = m_global_start_[i];
//		local_inner_count[i] = global_count[i];
//		local_inner_start[i] = m_global_start_[i];
//	}
//
//	if (num_process <= 1)
//		return;
//
//	int n = 0;
//	size_type L = 0;
//	for (int i = 0; i < m_ndims_; ++i)
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
//				+ m_global_start_[n];
//		local_inner_count[n] = (global_count[n] * (process_num + 1))
//				/ num_process + m_global_start_[n];
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
////		for (unsigned size_type s = 0, s_e = (1UL << (ndims_ * 2)); s < s_e; ++s)
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
////				bool id = f_inner && f_outer;
////
////				for (int i = 0; i < ndims_; ++i)
////				{
////					id = id && (remote.outer_count[i] != 0);
////				}
////				if (id)
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

//bool data_space::sync(std::shared_ptr<void> m_data, DataType const & DataType,
//		size_type id)
//{
//#if  !NO_MPI || USE_MPI
//	if (!GLOBAL_COMM.is_valid() || m_self_->send_recv_.size() == 0)
//	{
//		return true;
//	}
//
//	MPI_Comm comm = GLOBAL_COMM.comm();
//
//	MPI_Request request[m_self_->send_recv_.size() * 2];
//
//	int count = 0;
//
//	for (auto const & item : m_self_->send_recv_)
//	{
//
//		MPIDataType send_type = MPIDataType::clone(DataType, m_self_->local_shape_.m_ndims_ ,
//		&m_self_->local_shape_.dimensions[0], & item.send.m_global_start_[0],
//		&item.send.stride[0], &item.send.count[0], &item.send.block[0]);
//
//		dims_type recv_offset;
//		recv_offset = item.recv.m_global_start_ - m_self_->local_shape_.m_global_start_;
//
//		MPIDataType recv_type = MPIDataType::clone(DataType, m_self_->local_shape_.m_ndims_ ,
//		&m_self_->local_shape_.dimensions[0], & item.recv.m_global_start_[0],
//		&item.recv.stride[0], &item.recv.count[0], &item.recv.block[0]);
//
//		MPI_Isend(m_data.get(), 1, send_type.type(), item.dest, item.send_tag,
//		comm, &request[count * 2]);
//		MPI_Irecv(m_data.get(), 1, recv_type.type(), item.dest, item.recv_tag,
//		comm, &request[count * 2 + 1]);
//
//		++count;
//	}
//
//	MPI_Waitall(m_self_->send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);
//
//#endif //#if  !NO_MPI || USE_MPI
//
//	return true;
//}


}}//namespace simpla { namespace toolbox

