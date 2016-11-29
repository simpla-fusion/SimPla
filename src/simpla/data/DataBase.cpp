//
// Created by salmon on 16-11-9.
//
#include "DataBase.h"
#include "DataEntity.h"

namespace simpla { namespace data
{
struct DataBase::pimpl_s
{
    std::map<std::string, std::shared_ptr<DataBase> > m_table_;
};

DataBase::DataBase() : DataEntity(), m_pimpl_(new pimpl_s) {};

DataBase::~DataBase() {};

std::ostream &print_kv(std::ostream &os, int indent, std::string const &k, DataBase const &v)
{
    if (v.is_table()) { os << std::endl << std::setw(indent + 1) << " "; }
    os << k << " = " << v;
    return os;
}

std::ostream &DataBase::print(std::ostream &os, int indent) const
{
    if (!DataEntity::is_null()) { DataEntity::print(os, indent + 1); }

    if (!m_pimpl_->m_table_.empty())
    {
        auto it = m_pimpl_->m_table_.begin();
        auto ie = m_pimpl_->m_table_.end();
        if (it != ie)
        {
            os << "{ ";
            print_kv(os, indent, it->first, *it->second);
//            os << it->first << " = " << *it->second;
            ++it;
            for (; it != ie; ++it)
            {
                os << " , ";
                print_kv(os, indent, it->first, *it->second);
//                os << " , " << it->first << " = " << *it->second;
            }

            os << " }"  ;
        }
    }
    return os;
};

bool DataBase::is_table() const { return !m_pimpl_->m_table_.empty(); };

bool DataBase::is_null() const { return DataEntity::is_null() && m_pimpl_->m_table_.empty(); };

bool DataBase::empty() const { return DataEntity::empty() && m_pimpl_->m_table_.empty(); };

bool DataBase::has(std::string const &key) const { return m_pimpl_->m_table_.find(key) != m_pimpl_->m_table_.end(); };

void DataBase::insert(std::string const &key, std::shared_ptr<DataBase> const &v)
{
    m_pimpl_->m_table_.emplace(std::make_pair(key, v));
};

DataBase &DataBase::add(std::string const &key)
{
    auto res = clone();
    insert(key, res);
    return *res;
}

DataBase &DataBase::get(std::string const &key)
{
    if (key == "") { return *this; }

    auto it = m_pimpl_->m_table_.find(key);
    if (it == m_pimpl_->m_table_.end()) { return add(key); }
    else { return *(it->second); }

};

DataBase &DataBase::at(std::string const &key) { return key == "" ? *this : *(m_pimpl_->m_table_.at(key)); };

DataBase const &DataBase::at(std::string const &key) const
{
    return key == "" ? *this : *(m_pimpl_->m_table_.at(key));
};

//void DataBase::set_value(DataEntity const &other) { DataEntity(other).swap(m_pimpl_->m_value_); };
//
//void DataBase::set_value(DataEntity &&other) { other.swap(m_pimpl_->m_value_); };
//
//DataEntity &DataBase::get_value() { return m_pimpl_->m_value_; };
//
//DataEntity const &DataBase::get_value() const { return m_pimpl_->m_value_; };

bool DataBase::check(std::string const &key) { return has(key) && DataEntity::template as<bool>(); }

void
DataBase::foreach(std::function<void(std::string const &, DataBase const &)> const &fun) const
{
    for (auto &item: m_pimpl_->m_table_) { fun(item.first, *std::dynamic_pointer_cast<const DataBase>(item.second)); }
}

void
DataBase::foreach(std::function<void(std::string const &, DataBase &)> const &fun)
{
    for (auto &item: m_pimpl_->m_table_) { fun(item.first, *std::dynamic_pointer_cast<DataBase>(item.second)); }
};


}}//namespace simpla{namespace toolbox{