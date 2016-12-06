//
// Created by salmon on 16-11-9.
//
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Log.h>
#include "DataEntityTable.h"
#include "DataEntity.h"

namespace simpla { namespace data
{
struct DataEntityTable::pimpl_s
{
    std::map<std::string, std::shared_ptr<DataEntity> > m_table_;

    std::shared_ptr<DataEntity> &
    find_or_create(DataEntityTable *, std::string const &url, size_type start_pos = 0);

    DataEntity const *search(DataEntityTable const *, std::string const &url, size_type start_pos = 0);
};


std::shared_ptr<DataEntity> &
DataEntityTable::pimpl_s::find_or_create(DataEntityTable *t, std::string const &url, size_type start_pos)
{
    ASSERT (start_pos < url.size());
    ASSERT(t != nullptr);


    auto end_pos = url.find('.', start_pos);

    if (end_pos != std::string::npos)
    {
        std::string key = url.substr(start_pos, end_pos);

        auto it = t->m_pimpl_->m_table_.find(key);

        if (it == t->m_pimpl_->m_table_.end())
        {
            auto res = t->m_pimpl_->m_table_.emplace(std::make_pair(key, std::shared_ptr<DataEntityTable>()));

            return find_or_create(&res.first->second->as_table(), url, end_pos);

        } else if (it->second->is_table())
        {
            return find_or_create(&(it->second->as_table()), url, end_pos + 1);

        } else
        {

            RUNTIME_ERROR << "Try to insert an entity to non-table entity [" << url.substr(0, end_pos) << "]"
                          << std::endl;
        }
    } else
    {
        std::string key = url.substr(start_pos);

        auto it = t->m_pimpl_->m_table_.find(key);

        if (it == t->m_pimpl_->m_table_.end())
        {
            auto res = t->m_pimpl_->m_table_.emplace(std::make_pair(key, std::shared_ptr<DataEntityLight>()));

            return res.first->second;

        } else
        {
            return it->second;
        }
    }
};

DataEntity const *
DataEntityTable::pimpl_s::search(DataEntityTable const *t, std::string const &url, size_type start_pos)
{
    ASSERT (start_pos < url.size());
    ASSERT(t != nullptr);


    auto end_pos = url.find('.', start_pos);

    if (end_pos != std::string::npos)
    {
        std::string key = url.substr(start_pos, end_pos);

        auto it = t->m_pimpl_->m_table_.find(key);

        if (it != t->m_pimpl_->m_table_.end()) { return search(&(it->second->as_table()), url, end_pos + 1); }

    } else
    {
        std::string key = url.substr(start_pos);

        auto it = t->m_pimpl_->m_table_.find(key);

        if (it != t->m_pimpl_->m_table_.end()) { return it->second.get(); }
    }

    return nullptr;
};

DataEntityTable::DataEntityTable() : DataEntity(), m_pimpl_(new pimpl_s) {};

DataEntityTable::~DataEntityTable() {};

std::ostream &print_kv(std::ostream &os, int indent, std::string const &k, DataEntity const &v)
{
    if (v.is_table()) { os << std::endl << std::setw(indent + 1) << " "; }
    os << k << " = " << v;
    return os;
}

std::ostream &DataEntityTable::print(std::ostream &os, int indent) const
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

            os << " }";
        }
    }
    return os;
};


bool DataEntityTable::empty() const { return (m_pimpl_ != nullptr) && m_pimpl_->m_table_.empty(); };

bool DataEntityTable::has(std::string const &url) const { return m_pimpl_->search(this, url) != nullptr; };

bool DataEntityTable::check(std::string const &key)
{
    return has(key) && m_pimpl_->m_table_.at(key)->as_light().template as<bool>();
}

void DataEntityTable::set(std::string const &url, std::shared_ptr<DataEntity> const &v)
{
    std::shared_ptr<DataEntity>(v).swap(m_pimpl_->find_or_create(this, url));
};

std::shared_ptr<DataEntity> &DataEntityTable::get(std::string const &url)
{
    return m_pimpl_->find_or_create(this, url);
}


DataEntity &DataEntityTable::at(std::string const &url)
{
    auto res = m_pimpl_->search(this, url);
    if (res == nullptr) { OUT_OF_RANGE << "Can not find URL: [" << url << "] " << std::endl; }
    else { return *const_cast<DataEntity *>(res); }
};

DataEntity const &DataEntityTable::at(std::string const &url) const
{
    DataEntity const *res = m_pimpl_->search(this, url);
    if (res == nullptr) { OUT_OF_RANGE << "Can not find URL: [" << url << "] " << std::endl; } else { return *res; }
};


void
DataEntityTable::foreach(std::function<void(std::string const &, DataEntity const &)> const &fun) const
{
    for (auto &item: m_pimpl_->m_table_) { fun(item.first, *std::dynamic_pointer_cast<const DataEntity>(item.second)); }
}

void
DataEntityTable::foreach(std::function<void(std::string const &, DataEntity &)> const &fun)
{
    for (auto &item: m_pimpl_->m_table_) { fun(item.first, *item.second); }
};


}}//namespace simpla{namespace toolbox{