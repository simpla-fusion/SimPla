//
// Created by salmon on 16-10-8.
//
#include <ostream>
#include <iomanip>

#include "DataBase.h"
#include "DataEntity.h"

namespace simpla { namespace toolbox
{

std::ostream &operator<<(std::ostream &os, DataEntity const &prop) { return prop.print(os, 0); }

std::ostream &operator<<(std::ostream &os, DataBase const &prop) { return prop.print(os, 0); }

DataBase::DataBase() {};

DataBase::~DataBase() {};

std::ostream &DataBase::print(std::ostream &os, int indent) const
{
    if (m_value_ != nullptr) { m_value_->print(os, indent + 1); }

    if (!m_table_.empty())
    {
        auto it = m_table_.begin();
        auto ie = m_table_.end();
        if (it != ie)
        {
            os << "{ ";

            os << it->first << " = ";
            it->second->print(os, indent + 1);
            ++it;
            for (; it != ie; ++it)
            {
                os << "," << std::endl << std::setw(indent + 1) << "  " << it->first << " = ";
                it->second->print(os, indent + 2);

            }

            os << " }";
        }
    }
    return os;
};

void
DataBase::foreach(std::function<void(std::string const &, std::shared_ptr<DataBase> const &)> const &fun) const
{
    for (auto const &item:m_table_) { fun(item.first, item.second); }
}

void
DataBase::foreach(std::function<void(std::string const &, std::shared_ptr<DataBase> &)> const &fun)
{
    for (auto &item: m_table_) { fun(item.first, item.second); }
};


}}//namespace simpla{namespace toolbox{