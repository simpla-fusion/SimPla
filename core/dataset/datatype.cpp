/**
 * @file datatype.cpp
 *
 *  Created on: 2014-12-18
 *      Author: salmon
 */

#include "datatype.h"

#include <algorithm>
#include <iterator>

namespace simpla
{
struct DataType::pimpl_s
{
	pimpl_s();

	~pimpl_s();

	size_t m_ele_size_in_byte_ = 0;
	std::type_index m_t_index_;
	std::string m_name_;
	std::vector<size_t> m_extents_;

	std::vector<std::tuple<DataType, std::string, int>> m_members_;

};

DataType::DataType() :
		pimpl_(new pimpl_s)
{
}

DataType::pimpl_s::pimpl_s() :
		m_t_index_(std::type_index(typeid(void)))
{
}

DataType::pimpl_s::~pimpl_s()
{
}

DataType::DataType(std::type_index t_index, size_t ele_size_in_byte,
		int ndims, size_t const *dims, std::string name) :
		pimpl_(new pimpl_s)
{
	pimpl_->m_t_index_ = (t_index);
	pimpl_->m_ele_size_in_byte_ = (ele_size_in_byte);
	pimpl_->m_name_ = (name);

	if (ndims > 0 && dims != nullptr)
	{
		pimpl_->m_extents_.resize(ndims);
		for (int i = 0; i < ndims; ++i)
		{
			pimpl_->m_extents_[i] = dims[i];
		}
	}

	if (pimpl_->m_name_ == "")
	{
		if (is_same<int>())
		{ pimpl_->m_name_ = "int"; }
		else if (is_same<long>())
		{ pimpl_->m_name_ = "long"; }
		else if (is_same<unsigned long>())
		{ pimpl_->m_name_ = "unsigned long"; }
		else if (is_same<float>())
		{ pimpl_->m_name_ = "float"; }
		else if (is_same<double>())
		{ pimpl_->m_name_ = "double"; }
		else
		{ pimpl_->m_name_ = "UNKNOWN"; }
	}
}

DataType::DataType(const DataType &other) :
		pimpl_(new pimpl_s)
{
	pimpl_->m_ele_size_in_byte_ = (other.pimpl_->m_ele_size_in_byte_);
	pimpl_->m_t_index_ = (other.pimpl_->m_t_index_);
	pimpl_->m_name_ = (other.pimpl_->m_name_);

	std::copy(other.pimpl_->m_extents_.begin(), other.pimpl_->m_extents_.end(),
			std::back_inserter(pimpl_->m_extents_));
	std::copy(other.pimpl_->m_members_.begin(), other.pimpl_->m_members_.end(),
			std::back_inserter(pimpl_->m_members_));
}

DataType::~DataType()
{
}

DataType &DataType::operator=(DataType const &other)
{
	pimpl_->m_ele_size_in_byte_ = (other.pimpl_->m_ele_size_in_byte_);
	pimpl_->m_t_index_ = (other.pimpl_->m_t_index_);
	pimpl_->m_name_ = (other.pimpl_->m_name_);

	std::copy(other.pimpl_->m_extents_.begin(), other.pimpl_->m_extents_.end(),
			std::back_inserter(pimpl_->m_extents_));
	std::copy(other.pimpl_->m_members_.begin(), other.pimpl_->m_members_.end(),
			std::back_inserter(pimpl_->m_members_));

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
	return std::move(pimpl_->m_t_index_.name());
}

bool DataType::is_valid() const
{
	return pimpl_->m_t_index_ != std::type_index(typeid(void));
}

size_t DataType::ele_size_in_byte() const
{
	return pimpl_->m_ele_size_in_byte_;
}

size_t DataType::size() const
{
	size_t res = 1;

	for (auto const &d : pimpl_->m_extents_)
	{
		res *= d;
	}
	return res;
}

size_t DataType::size_in_byte() const
{
	return pimpl_->m_ele_size_in_byte_ * size();
}

int DataType::rank() const
{
	return static_cast<int>(pimpl_->m_extents_.size());
}

size_t DataType::extent(size_t n) const
{
	return pimpl_->m_extents_[n];
}

std::vector<size_t> const &DataType::extents() const
{
	return pimpl_->m_extents_;
}

void DataType::extent(size_t *d) const
{
	std::copy(pimpl_->m_extents_.begin(), pimpl_->m_extents_.end(), d);
}

void DataType::extent(int rank, size_t const *d)
{
	pimpl_->m_extents_.resize(rank);
	for (int i = 0; i < rank; ++i)
	{
		pimpl_->m_extents_[i] = d[i];
	}

}

std::vector<std::tuple<DataType, std::string, int>> const &DataType::members() const
{
	return pimpl_->m_members_;
}

bool DataType::is_compound() const
{
	return pimpl_->m_members_.size() > 0;
}

bool DataType::is_array() const
{
	return pimpl_->m_extents_.size() > 0;
}

bool DataType::is_opaque() const
{
	return pimpl_->m_extents_.size() == 0
			&& pimpl_->m_t_index_ == std::type_index(typeid(void));
}

bool DataType::is_same(std::type_index const &other) const
{
	return pimpl_->m_t_index_ == other;
}

void DataType::push_back(DataType &&d_type, std::string const &name, int pos)
{
	if (pos < 0)
	{
		if (pimpl_->m_members_.empty())
		{
			pos = 0;
		}
		else
		{
			pos =
					std::get<2>(*(pimpl_->m_members_.rbegin()))
							+ std::get<0>(*(pimpl_->m_members_.rbegin())).size_in_byte();
		}
	}

	pimpl_->m_members_.push_back(std::forward_as_tuple(d_type, name, pos));

}

namespace traits
{
std::ostream &print(std::ostream &os, DataType const &self)
{

	if (self.is_compound())
	{
		os << "DATATYPE" << std::endl <<

				"struct " << self.name() << std::endl

				<< "{" << std::endl;

		auto it = self.members().begin();
		auto ie = self.members().end();

		os << "\t";
		print(os, std::get<0>(*it));
		os << "\t" << std::get<1>(*it);

		++it;

		for (; it != ie; ++it)
		{
			os << "," << std::endl << "\t";

			print(os, std::get<0>(*it));
			os << "\t" << std::get<1>(*it);
		}
		os << std::endl << "};" << std::endl;
	}
	else
	{
		os << self.name();
		for (auto const &d : self.extents())
		{
			os << "[" << d << "]";
		}
	}

	return os;
}
} //namespace traits

}  // namespace simpla
