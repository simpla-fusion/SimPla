//
// Created by salmon on 16-10-7.
//


#include <iomanip>
#include "DataBaseAny.h"

namespace simpla { namespace toolbox
{

void DataBaseAny::swap(DataBaseAny &other)
{
    m_value_.swap(other.m_value_);
    m_table_.swap(other.m_table_);
}


size_t DataBaseAny::size() const { return m_table_.size(); }

bool DataBaseAny::empty() const { return m_table_.empty(); }

bool DataBaseAny::has(std::string const &key) const { return m_table_.find(key) != m_table_.end(); }

DataBaseAny::iterator DataBaseAny::find(std::string const &key) { return m_table_.find(key); };

std::pair<DataBaseAny::iterator, bool> DataBaseAny::insert(std::string const &key, std::shared_ptr<DataBase> &v)
{
    auto res = m_table_.emplace(key, v);
    if (res.second)
    {
        return std::make_pair(DataBaseAny::iterator(res.first), true);

    } else
    {
        return std::make_pair(DataBaseAny::iterator(), false);
    }
};

std::shared_ptr<DataBase> DataBaseAny::get(std::string const &key)
{
    std::shared_ptr<DataBase> res;
    auto it = m_table_.find(key);
    if (it == m_table_.end()) { res = std::shared_ptr<DataBase>(new DataBaseAny); } else { return it->second; }
}

std::shared_ptr<DataBase> DataBaseAny::at(std::string const &key)
{
    auto it = m_table_.find(key);
    if (it == m_table_.end()) { return std::shared_ptr<DataBase>(nullptr); } else { return it->second; }
}


std::shared_ptr<const DataBase> DataBaseAny::at(std::string const &key) const
{
    auto it = m_table_.find(key);
    if (it == m_table_.end()) { return std::shared_ptr<DataBase>(nullptr); } else { return it->second; }
};


//DataBaseAny::iterator DataBaseAny::begin() { return DataBaseAny::iterator(m_table_.begin()); }
//
//DataBaseAny::iterator DataBaseAny::end() { return DataBaseAny::iterator(m_table_.end()); }
//
//DataBaseAny::iterator DataBaseAny::begin() const { return DataBaseAny::iterator(m_table_.begin()); }
//
//DataBaseAny::iterator DataBaseAny::end() const { return DataBaseAny::iterator(m_table_.end()); }
bool DataBaseAny::open(std::string path) {}

void DataBaseAny::close() {}

std::ostream &DataBaseAny::print(std::ostream &os, int indent) const
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
