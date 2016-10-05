/**
 * @file WriteBuffer.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_WRITEBUFFER_H
#define SIMPLA_WRITEBUFFER_H

#include "../DataType.h"
#include "../DataSpace.h"
#include "../../toolbox/HDF5Stream.h"

namespace simpla { namespace io
{
class WriteBuffer
{
    data_model::DataType m_data_type_;
    size_t m_record_size_ = 1;
    size_t m_max_buffer_depth_ = 1;
    size_t m_buffer_depth_ = 0;

    std::string m_url_;

    std::shared_ptr<void> m_buffer_;

    HDF5Stream &m_out_stream_;
public:
    WriteBuffer(HDF5Stream &);

    ~WriteBuffer();

    bool empty() const { return m_buffer_depth_ == 0; }

    void flush();

    void data_type(data_model::DataType const &d_type) { m_data_type_ = d_type; }

    void record_size(size_t s) { m_record_size_ = s; }

    void max_buffer_depth(size_t s) { m_max_buffer_depth_ = s; }

    void *buffer();

    std::string write(bool is_flush = false);

};

template<typename T>
class WriteBufferProxy : public WriteBuffer
{
    typedef T value_type;
public:
    WriteBufferProxy(HDF5Stream &os) : WriteBuffer(os) { WriteBuffer::data_type(data_model::DataType::create<T>()); }

    virtual ~ WriteBufferProxy() { }


    template<typename TI>
    std::string write(TI const &b, TI const &e)
    {
        std::copy(b, e, reinterpret_cast<value_type *>(WriteBuffer::buffer()));
        return WriteBuffer::write();
    }


};
}}//namespace simpla { namespace io

#endif //SIMPLA_WRITEBUFFER_H
