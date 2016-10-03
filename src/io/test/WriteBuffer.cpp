/**
 * @file WriteBuffer.cpp
 * @author salmon
 * @date 2016-01-13.
 */

#include "WriteBuffer.h"
#include "../HDF5Stream.h"
#include "../toolbox/utilities/MemoryPool.h"

namespace simpla { namespace io
{

void *WriteBuffer::buffer()
{
    ASSERT (m_buffer_depth_ < m_max_buffer_depth_)

    if (m_buffer_ == nullptr)
    {
        m_buffer_ = sp_alloc_memory(m_data_type_.size() * m_record_size_ * m_max_buffer_depth_);
    }
    ++m_buffer_depth_;
    return reinterpret_cast<void *>(reinterpret_cast<char *>(m_buffer_.get()) + (m_buffer_depth_ - 1) * m_record_size_);
}

std::string WriteBuffer::write(bool is_flush)
{
    if (m_buffer_depth_ >= m_max_buffer_depth_ || is_flush)
    {
        size_t dims[2] = {m_buffer_depth_, m_record_size_};
        m_buffer_depth_ = 0;
        data_model::DataSet dset;

        dset.data_type = m_data_type_;
        dset.data_space = data_model::DataSpace::create_simple(2, dims);
        dset.memory_space = dset.data_space;
        dset.data = m_buffer_;


        return m_out_stream_.write(m_url_, dset, SP_APPEND);

    }
    else
    {
        return "";
    }


}

void WriteBuffer::flush() { write(true); }

WriteBuffer::WriteBuffer(IOStream &os) : m_out_stream_(os) { }

WriteBuffer::~WriteBuffer() { flush(); }
}}