//
// Created by salmon on 16-11-9.
//
#include "DataEntityTable.h"
#include "DataEntity.h"

namespace simpla { namespace data
{
struct DataEntityTable::pimpl_s
{
    std::map<std::string, std::shared_ptr<DataEntity> > m_table_;

    std::shared_ptr<DataEntityTable>
    parse_url(DataEntityTable &, std::string const &url, size_type start_pos = 0);
};


std::shared_ptr<DataEntity>
get_url(DataEntityTable &t, std::string const &url, size_type start_pos)
{
    std::shared_ptr<DataEntityTable> res(nullptr);


    auto end_pos = url.find('.', start_pos, 1);

    if (end_pos != std::string::npos)
    {
        std::string key = url.substr(start_pos, end_pos);

        auto it = m_table_.find(key);

        if (it != m_table_.end()) { res = it->second; }
        else
        {
            res = std::make_shared<DataEntityTable>();
            m_table_.emplace(std::make_pair(key, res));
        }

        res = parse_url(*res, url, end_pos);
    } else
    {

    }
    return res;
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

bool DataEntityTable::has(std::string const &key) const
{
    return m_pimpl_->m_table_.find(key) != m_pimpl_->m_table_.end();
};

void DataEntityTable::insert(std::string const &key, std::shared_ptr<DataEntity> const &v)
{
    m_pimpl_->m_table_.emplace(std::make_pair(key, v));
};

DataEntity &DataEntityTable::add(std::string const &key)
{
    auto res = std::make_shared<DataEntityLight>();
    insert(key, res);
    return *res;
}

DataEntity &DataEntityTable::get(std::string const &key)
{
    if (key == "") { return *this; }

    auto it = m_pimpl_->m_table_.find(key);
    if (it == m_pimpl_->m_table_.end()) { return add(key); }
    else { return *(it->second); }

};

DataEntity &DataEntityTable::at(std::string const &key) { return *(m_pimpl_->m_table_.at(key)); };

DataEntity const &DataEntityTable::at(std::string const &key) const { return *(m_pimpl_->m_table_.at(key)); };

bool DataEntityTable::check(std::string const &key)
{
    return has(key) && m_pimpl_->m_table_.at(key)->as_light().template as<bool>();
}

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