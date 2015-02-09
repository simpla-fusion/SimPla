/**
 * @file data_type.cpp
 *
 *  Created on: 2014年12月18日
 *      Author: salmon
 */

#include "datatype.h"

namespace simpla
{
struct DataType::pimpl_s
{
	pimpl_s();
	~pimpl_s();
	size_t ele_size_in_byte_ = 0;
	std::type_index t_index_;
	std::string name_;
	std::vector<size_t> extents_;

	std::vector<std::tuple<DataType, std::string, int>> members_;

};
DataType::DataType() :
		pimpl_(new pimpl_s)
{
}
DataType::pimpl_s::pimpl_s() :
		t_index_(std::type_index(typeid(void)))
{
}
DataType::pimpl_s::~pimpl_s()
{
}
DataType::DataType(std::type_index t_index, size_t ele_size_in_byte,
		unsigned int ndims, size_t* dims, std::string name) :
		pimpl_(new pimpl_s)
{
	pimpl_->t_index_ = (t_index);
	pimpl_->ele_size_in_byte_ = (ele_size_in_byte);
	pimpl_->name_ = (name);

	if (ndims > 0 && dims != nullptr)
	{
		pimpl_->extents_.resize(ndims);
		for (int i = 0; i < ndims; ++i)
		{
			pimpl_->extents_[i] = dims[i];
		}
	}

	if (pimpl_->name_ == "")
	{
		if (is_same<int>())
		{
			pimpl_->name_ = "int";
		}
		else if (is_same<long>())
		{
			pimpl_->name_ = "long";
		}
		else if (is_same<unsigned long>())
		{
			pimpl_->name_ = "unsigned long";
		}
		else if (is_same<float>())
		{
			pimpl_->name_ = "float";
		}
		else if (is_same<double>())
		{
			pimpl_->name_ = "double";
		}
		else
		{
			pimpl_->name_ = "UNKNOWN";
		}
	}
}

DataType::DataType(const DataType & other) :
		pimpl_(new pimpl_s)
{
	pimpl_->ele_size_in_byte_ = (other.pimpl_->ele_size_in_byte_);
	pimpl_->t_index_ = (other.pimpl_->t_index_);
	pimpl_->name_ = (other.pimpl_->name_);

	std::copy(other.pimpl_->extents_.begin(), other.pimpl_->extents_.end(),
			std::back_inserter(pimpl_->extents_));
	std::copy(other.pimpl_->members_.begin(), other.pimpl_->members_.end(),
			std::back_inserter(pimpl_->members_));
}
DataType::DataType(DataType && other) :
		pimpl_(other.pimpl_)
{
	other.pimpl_ = nullptr;
//		std::copy(other.extents_.begin(), other.extents_.end(),
//				std::back_inserter(extents_));
//		std::copy(other.data.begin(), other.data.end(),
//				std::back_inserter(data));
}

DataType::~DataType()
{
	if (pimpl_ != nullptr)
		delete pimpl_;
}

DataType& DataType::operator=(DataType const& other)
{
	pimpl_->ele_size_in_byte_ = (other.pimpl_->ele_size_in_byte_);
	pimpl_->t_index_ = (other.pimpl_->t_index_);
	pimpl_->name_ = (other.pimpl_->name_);

	std::copy(other.pimpl_->extents_.begin(), other.pimpl_->extents_.end(),
			std::back_inserter(pimpl_->extents_));
	std::copy(other.pimpl_->members_.begin(), other.pimpl_->members_.end(),
			std::back_inserter(pimpl_->members_));

	return *this;
}
void DataType::swap(DataType & other)
{
	std::swap(pimpl_, other.pimpl_);
}

std::string DataType::name() const
{
	return std::move(pimpl_->t_index_.name());
}
bool DataType::is_valid() const
{
	return pimpl_->t_index_ != std::type_index(typeid(void));
}
size_t DataType::ele_size_in_byte() const
{
	return pimpl_->ele_size_in_byte_;
}

size_t DataType::size() const
{
	size_t res = 1;

	for (auto const & d : pimpl_->extents_)
	{
		res *= d;
	}
	return res;
}
size_t DataType::size_in_byte() const
{
	return pimpl_->ele_size_in_byte_ * size();
}
size_t DataType::rank() const
{
	return pimpl_->extents_.size();
}
size_t DataType::extent(size_t n) const
{
	return pimpl_->extents_[n];
}
void DataType::extent(size_t rank, size_t const*d)
{
	pimpl_->extents_.resize(rank);
	for (int i = 0; i < rank; ++i)
	{
		pimpl_->extents_[i] = d[i];
	}

}
std::vector<std::tuple<DataType, std::string, int>> const & DataType::members() const
{
	return pimpl_->members_;
}
bool DataType::is_compound() const
{
	return pimpl_->members_.size() > 0;
}
bool DataType::is_array() const
{
	return pimpl_->extents_.size() > 0;
}
bool DataType::is_opaque() const
{
	return pimpl_->extents_.size() == 0
			&& pimpl_->t_index_ == std::type_index(typeid(void));
}

bool DataType::is_same(std::type_index const &other) const
{
	return pimpl_->t_index_ == other;
}

void DataType::push_back(DataType && d_type, std::string const & name, int pos)
{
	if (pos < 0)
	{
		if (pimpl_->members_.empty())
		{
			pos = 0;
		}
		else
		{
			pos = std::get<2>(*(pimpl_->members_.rbegin()))
					+ std::get<0>(*(pimpl_->members_.rbegin())).size_in_byte();
		}
	}

	pimpl_->members_.push_back(std::forward_as_tuple(d_type, name, pos));

}
std::ostream & DataType::print(std::ostream & os) const
{

	if (is_compound())
	{
		os << "DATATYPE" << std::endl <<

		"struct " << pimpl_->name_ << std::endl

		<< "{" << std::endl;

		auto it = pimpl_->members_.begin();
		auto ie = pimpl_->members_.end();

		os << "\t";
		std::get<0>(*it).print(os);
		os << "\t" << std::get<1>(*it);

		++it;

		for (; it != ie; ++it)
		{
			os << "," << std::endl << "\t";
			std::get<0>(*it).print(os);
			os << "\t" << std::get<1>(*it);
		}
		os << std::endl << "};" << std::endl;
	}
	else
	{
		os << pimpl_->name_;
		for (auto const & d : pimpl_->extents_)
		{
			os << "[" << d << "]";
		}
	}

	return os;
}

}  // namespace simpla
