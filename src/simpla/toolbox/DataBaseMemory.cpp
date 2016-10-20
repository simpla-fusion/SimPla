//
// Created by salmon on 16-10-7.
//


#include <iomanip>
#include <assert.h>
#include "DataBaseMemory.h"

namespace simpla { namespace toolbox
{

void DataBaseMemory::swap(DataBaseMemory &other) { m_table_.swap(other.m_table_); }

size_t DataBaseMemory::size() const { return m_table_.size(); }

bool DataBaseMemory::empty() const { return m_table_.empty(); }

bool DataBaseMemory::has(std::string const &key) const { return m_table_.find(key) != m_table_.end(); }

void DataBaseMemory::set(std::string const &key, std::shared_ptr<DataEntity> const &v)
{
    m_table_[key] = v;
};

void DataBaseMemory::set(std::string const &key, std::shared_ptr<DataEntityAny> const &v)
{
    set(key, std::dynamic_pointer_cast<DataEntity>(v));
};

void DataBaseMemory::set(std::string const &key, std::shared_ptr<DataBaseMemory> const &v)
{
    set(key, std::dynamic_pointer_cast<DataEntity>(v));
};

std::shared_ptr<DataEntity> DataBaseMemory::get(std::string const &key) { return m_table_[key]; }

std::shared_ptr<DataEntity> DataBaseMemory::at(std::string const &key) { return m_table_.at(key); }

//std::shared_ptr<DataBase> DataBaseMemory::sub(std::string const &key)
//{
//    auto res = m_table_.at(key);
//    assert(res->is_table());
//    return std::dynamic_pointer_cast<DataBase>(res);
//};
//
//std::shared_ptr<const DataBase> DataBaseMemory::sub(std::string const &key) const
//{
//    auto res = m_table_.at(key);
//    assert(res->is_table());
//    return std::dynamic_pointer_cast<const DataBase>(res);
//};


std::shared_ptr<const DataEntity> DataBaseMemory::at(std::string const &key) const { return m_table_.at(key); }

void DataBaseMemory::foreach(std::function<void(std::string const &, DataEntity &)> const &fun)
{

    for (auto &item:m_table_)
    {
        fun(item.first, *std::dynamic_pointer_cast<DataEntity>(item.second));
    }
};

void DataBaseMemory::foreach(std::function<void(std::string const &, DataEntity const &)> const &fun) const
{

    for (auto const &item:m_table_)
    {
        fun(item.first, *std::dynamic_pointer_cast<const DataEntity>(item.second));
    }
};


}}//namespace simpla{namespace toolbox{
