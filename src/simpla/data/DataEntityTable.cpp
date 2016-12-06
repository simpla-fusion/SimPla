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

    std::shared_ptr<DataEntity> &emplace(DataEntityTable *t, std::string const &url, char split_char = '.');

    DataEntity const *search(DataEntityTable const *, std::string const &url, char split_char = '.');
};

std::shared_ptr<DataEntity> &
DataEntityTable::pimpl_s::emplace(DataEntityTable *t, std::string const &url, char split_char)
{
    size_type start_pos = 0;
    size_type end_pos = url.size();
    while (start_pos < end_pos)
    {
        size_type pos = url.find(split_char, start_pos);

        if (pos != std::string::npos)
        {
            auto res = t->m_pimpl_->m_table_.emplace(
                    url.substr(start_pos, pos - start_pos),
                    std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityTable>()));

            if (!res.first->second->is_table()) { break; }
            else if (pos == end_pos - 1) { return res.first->second; }

            t = &res.first->second.get()->as_table();
            start_pos = pos + 1;
            continue;

        } else
        {
            auto res = t->m_pimpl_->m_table_.emplace(url.substr(start_pos), std::make_shared<DataEntityLight>());

            return res.first->second;

        }
    }
    RUNTIME_ERROR << " Can not insert entity at [" << url << "]" << std::endl;
};


DataEntity const *
DataEntityTable::pimpl_s::search(DataEntityTable const *t, std::string const &url, char split_char)
{
    size_type start_pos = 0;
    size_type end_pos = url.size();
    while (start_pos < end_pos)
    {
        size_type pos = url.find(split_char, start_pos);

        if (pos != std::string::npos)
        {
            auto it = t->m_pimpl_->m_table_.find(url.substr(start_pos, pos - start_pos));

            if (pos == end_pos - 1) { return it->second.get(); }
            else if (it == t->m_pimpl_->m_table_.end() || !it->second->is_table()) { break; }

            t = &it->second->as_table();
            start_pos = pos + 1;
            continue;

        } else
        {
            auto it = t->m_pimpl_->m_table_.find(url.substr(start_pos));

            if (it != t->m_pimpl_->m_table_.end()) { return it->second.get(); }

            break;
        }
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


bool
DataEntityTable::empty() const { return (m_pimpl_ != nullptr) && m_pimpl_->m_table_.empty(); };

bool
DataEntityTable::has(std::string const &url) const { return m_pimpl_->search(this, url) != nullptr; };

bool
DataEntityTable::check(std::string const &key) const
{
    return has(key) && m_pimpl_->m_table_.at(key)->as_light().template as<bool>();
}

DataEntityTable *
DataEntityTable::create_table(std::string const &url) { return &m_pimpl_->emplace(this, url + ".")->as_table(); }

void DataEntityTable::set(std::string const &url, std::shared_ptr<DataEntity> const &v)
{
    std::shared_ptr<DataEntity>(v).swap(m_pimpl_->emplace(this, url));
};

std::shared_ptr<DataEntity> &DataEntityTable::get(std::string const &url)
{
    return m_pimpl_->emplace(this, url);
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
    if (res == nullptr) { throw std::out_of_range("Can not find URL: [" + url + "] "); } else { return *res; }
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