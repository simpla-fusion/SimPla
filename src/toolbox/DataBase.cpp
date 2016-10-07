//
// Created by salmon on 16-10-7.
//

#include <iomanip>
#include "DataBase.h"

namespace simpla { namespace toolbox
{

void DataBase::swap(DataBase &other)
{
    m_value_.swap(other.m_value_);
    m_table_.swap(other.m_table_);
}


size_t DataBase::size() const { return m_table_.size(); }

bool DataBase::empty() const { return m_table_.empty(); }

bool DataBase::has_a(std::string const &key) const { return m_table_.find(key) != m_table_.end(); }


std::pair<DataBase::iterator, bool> DataBase::insert(std::string const &key, std::shared_ptr<DataBase> &v)
{
    auto res = m_table_.emplace(key, v);
    if (res.second)
    {
        return std::make_pair(DataBase::iterator(res.first), true);

    } else
    {
        return std::make_pair(DataBase::iterator(), false);
    }
};

std::shared_ptr<DataBase> DataBase::at(std::string const &key)
{
    auto it = m_table_.find(key);
    if (it == m_table_.end()) { return std::shared_ptr<DataBase>(nullptr); } else { return it->second; }
}


std::shared_ptr<const DataBase> DataBase::at(std::string const &key) const
{
    auto it = m_table_.find(key);
    if (it == m_table_.end()) { return std::shared_ptr<DataBase>(nullptr); } else { return it->second; }
};


//DataBase::iterator DataBase::begin() { return DataBase::iterator(m_table_.begin()); }
//
//DataBase::iterator DataBase::end() { return DataBase::iterator(m_table_.end()); }
//
//DataBase::iterator DataBase::begin() const { return DataBase::iterator(m_table_.begin()); }
//
//DataBase::iterator DataBase::end() const { return DataBase::iterator(m_table_.end()); }

std::ostream &DataBase::print(std::ostream &os, int indent) const
{
    if (!empty())
    {
        auto it = this->begin();
        auto ie = this->end();


        os << std::endl << std::setw(indent + 1) << "{" << std::endl;


        os << std::setw(indent + 1) << " " << it->first << " = ";
        it->second->print(os, indent + 1);
        ++it;

        for (; it != ie; ++it)
        {
            os << " , " << std::endl << std::setw(indent + 1) << " " << it->first << " = ";
            it->second->print(os, indent + 1);
        }
        os << std::endl << std::setw(indent + 1) << "}";

    }

    return os;


};

}}//namespace simpla{namespace toolbox{
