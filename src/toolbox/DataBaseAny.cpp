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

void DataBaseAny::set(std::string const &key, std::shared_ptr<DataBase> const &v)
{
    if (v->is_a(typeid(DataBaseAny))) { set(key, std::dynamic_pointer_cast<DataBaseAny>(v)); }
    else { UNIMPLEMENTED; }

};

void DataBaseAny::set(std::string const &key, std::shared_ptr<DataBaseAny> const &v)
{
    m_table_[key] = std::dynamic_pointer_cast<DataBaseAny>(v);
};

//DataBase::iterator DataBaseAny::find(std::string const &key) { return m_table_.find(key); };

//std::pair<DataBaseAny::iterator, bool> DataBaseAny::set(std::string const &key, std::shared_ptr<DataBase> &v)
//{
//    auto res = m_table_.emplace(key, v);
//    if (res.second)
//    {
//        return std::make_pair(DataBaseAny::iterator(res.first), true);
//
//    } else
//    {
//        return std::make_pair(DataBaseAny::iterator(), false);
//    }
//};

std::shared_ptr<DataBase> DataBaseAny::get(std::string const &key) { return m_table_[key]; }

std::shared_ptr<DataBase> DataBaseAny::at(std::string const &key) { return m_table_.at(key); }


std::shared_ptr<const DataBase> DataBaseAny::at(std::string const &key) const { return m_table_.at(key); }

void DataBaseAny::for_each(std::function<void(std::string const &, DataBase &)> const &fun)
{

    for (auto &item:m_table_)
    {
        fun(item.first, *std::dynamic_pointer_cast<DataBase>(item.second));
    }
};

void DataBaseAny::for_each(std::function<void(std::string const &, DataBase const &)> const &fun) const
{

    for (auto const &item:m_table_)
    {
        fun(item.first, *std::dynamic_pointer_cast<const DataBase>(item.second));
    }
};


}}//namespace simpla{namespace toolbox{
