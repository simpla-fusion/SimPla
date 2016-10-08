/**
 * @file DataType.cpp
 *
 *  Created on: 2014-12-18
 *      Author: salmon
 */

#include "DataType.h"
#include "SIMPLA_config.h"

#include <algorithm>
#include <iterator>
#include <iomanip>

namespace simpla { namespace toolbox
{
struct DataType::pimpl_s
{
    pimpl_s();

    pimpl_s(pimpl_s const &);

    pimpl_s(pimpl_s &&);

    ~pimpl_s();

    size_type m_size_in_byte_ = 0;
    size_type m_ele_size_in_byte_ = 0;
    std::type_index m_t_index_;
    std::string m_name_;
    std::vector<size_t> m_extents_;

    std::vector<std::tuple<DataType, std::string, int>> m_members_;

};

DataType::DataType()
        : pimpl_(new pimpl_s) {}

DataType::DataType(DataType const &other)
        : pimpl_(new pimpl_s(*other.pimpl_)) {}

DataType::DataType(DataType &&other)
        : pimpl_(new pimpl_s(*other.pimpl_)) {}

DataType::~DataType() {}

DataType::pimpl_s::pimpl_s()
        : m_t_index_(std::type_index(typeid(void))) {}

DataType::pimpl_s::pimpl_s(pimpl_s const &other)
        : m_size_in_byte_(other.m_size_in_byte_),
          m_ele_size_in_byte_(other.m_ele_size_in_byte_),
          m_t_index_(other.m_t_index_),
          m_name_(other.m_name_),
          m_extents_(other.m_extents_),
          m_members_(other.m_members_) {}

DataType::pimpl_s::pimpl_s(pimpl_s &&other)
        : m_size_in_byte_(other.m_size_in_byte_),
          m_ele_size_in_byte_(other.m_ele_size_in_byte_),
          m_t_index_(other.m_t_index_),
          m_name_(other.m_name_),
          m_extents_(other.m_extents_),
          m_members_(other.m_members_) {}

DataType::pimpl_s::~pimpl_s() {}

DataType::DataType(std::type_index t_index, size_type ele_size_in_byte,
                   int ndims, size_type const *dims, std::string name)
        : pimpl_(new pimpl_s)
{
    pimpl_->m_t_index_ = (t_index);
    pimpl_->m_ele_size_in_byte_ = (ele_size_in_byte);
    pimpl_->m_name_ = (name);
    pimpl_->m_size_in_byte_ = (ele_size_in_byte);

    if (ndims > 0 && dims != nullptr)
    {
        pimpl_->m_extents_.resize(ndims);

        for (int i = 0; i < ndims; ++i)
        {
            pimpl_->m_extents_[i] = dims[i];
            pimpl_->m_size_in_byte_ *= dims[i];
        }
    }


    if (pimpl_->m_name_ == "")
    {
        if (is_same<int>()) { pimpl_->m_name_ = "int"; }
        else if (is_same<long>()) { pimpl_->m_name_ = "long"; }
        else if (is_same<unsigned long>()) { pimpl_->m_name_ = "unsigned long"; }
        else if (is_same<float>()) { pimpl_->m_name_ = "float"; }
        else if (is_same<double>()) { pimpl_->m_name_ = "double"; }
        else { pimpl_->m_name_ = "UNKNOWN"; }
    }
}

DataType &DataType::operator=(DataType const &other)
{
//    m_self_->m_ele_size_in_byte_ = (other.m_self_->m_ele_size_in_byte_);
//    m_self_->m_t_index_ = (other.m_self_->m_t_index_);
//    m_self_->m_name_ = (other.m_self_->m_name_);
//
//    std::copy(other.m_self_->m_extents_.begin(), other.m_self_->m_extents_.end(),
//              std::back_inserter(m_self_->m_extents_));
//    std::copy(other.m_self_->m_members_.begin(), other.m_self_->m_members_.end(),
//              std::back_inserter(m_self_->m_members_));

    DataType(other).swap(*this);

    return *this;
}

void DataType::swap(DataType &other)
{
    std::swap(pimpl_, other.pimpl_);
}

DataType DataType::element_type() const
{
    DataType res(*this);

    res.pimpl_->m_extents_.clear();

    return std::move(res);
}

std::string DataType::name() const
{
    return std::move(pimpl_->m_name_);
}

bool DataType::is_valid() const
{
    return pimpl_ != nullptr;
//    return pimpl_->m_t_index_ != std::type_index(typeid(void));
}

size_type DataType::ele_size_in_byte() const
{
    return pimpl_->m_ele_size_in_byte_;
}

size_type DataType::number_of_entities() const
{
    size_type res = 1;
    for (auto const &d : pimpl_->m_extents_) { res *= d; }

    return res;
}

size_type DataType::size_in_byte() const { return pimpl_->m_size_in_byte_; }

void DataType::size_in_byte(size_type s) { pimpl_->m_size_in_byte_ = s; };

int DataType::rank() const { return static_cast<int>(pimpl_->m_extents_.size()); }

size_type DataType::extent(int n) const { return pimpl_->m_extents_[n]; }

std::vector<size_t> const &DataType::extents() const { return pimpl_->m_extents_; }

void DataType::extent(size_type *d) const
{
    std::copy(pimpl_->m_extents_.begin(), pimpl_->m_extents_.end(), d);
}

void DataType::extent(int rank, const size_type *d)
{
    pimpl_->m_extents_.resize(rank);
    for (int i = 0; i < rank; ++i)
    {
        pimpl_->m_extents_[i] = d[i];
    }
}

std::vector<std::tuple<DataType, std::string, int>> const &DataType::members() const { return pimpl_->m_members_; }

bool DataType::is_compound() const { return pimpl_->m_members_.size() > 0; }

bool DataType::is_array() const { return pimpl_->m_extents_.size() > 0; }

bool DataType::is_opaque() const
{
    return pimpl_->m_extents_.size() == 0
           && pimpl_->m_t_index_ == std::type_index(typeid(void));
}

bool DataType::is_same(std::type_index const &other) const { return pimpl_->m_t_index_ == other; }

int DataType::push_back(DataType const &d_type, std::string const &name, size_type offset)
{
    if (offset < 0)
    {
        if (pimpl_->m_members_.empty())
        {
            offset = 0;
        } else
        {
            offset = (std::get<2>(*(pimpl_->m_members_.rbegin()))
                      + std::get<0>(*(pimpl_->m_members_.rbegin())).size_in_byte());
        }
    }

    pimpl_->m_members_.push_back(std::forward_as_tuple(d_type, name, offset));

    return SP_SUCCESS;

}

std::ostream &DataType::print(std::ostream &os, int indent) const
{

    if (this->is_compound())
    {
        os << std::setw(indent) << "DATATYPE" << std::endl << std::setw(indent)

           << "struct " << this->name() << std::endl << std::setw(indent)

           << "{" << std::endl << std::setw(indent);

        auto it = this->members().begin();
        auto ie = this->members().end();

        os << "\t";
        std::get<0>(*it).print(os, indent + 1);
        os << "\t" << std::get<1>(*it);

        ++it;

        for (; it != ie; ++it)
        {
            os << "," << std::endl << "\t";
            std::get<0>(*it).print(os, indent + 1);
            os << "\t" << std::get<1>(*it);
        }
        os << std::endl << std::setw(indent)
           << "};" << std::endl << std::setw(indent);
    } else
    {
        os << this->name();
        for (auto const &d : this->extents())
        {
            os << "[" << d << "]";
        }
    }

    return os;
}

}} //namespace simpla { namespace data_model

