//
// Created by salmon on 17-8-18.
//

#include "DataNode.h"
#include <iomanip>
#include <map>
#include <vector>
#include "DataEntity.h"

#include "DataArray.h"
#include "DataBase.h"
#include "DataTable.h"
namespace simpla {
namespace data {
struct pimpl_s {
    std::map<std::string, std::shared_ptr<DataEntity>> m_attributes_;
    std::vector<std::shared_ptr<DataNode>> m_children_;
};
DataNode::DataNode() = default;
DataNode::~DataNode() {}
std::shared_ptr<DataNode> DataNode::New(std::string const& s) {
    return s.empty() ? std::shared_ptr<DataNode>(new DataNode()) : data::DataBase::New(s)->Root();
}

std::ostream& Print(std::ostream& os, std::shared_ptr<DataEntity> const& v, int indent = 0) { return os; }

std::ostream& Print(std::ostream& os, std::shared_ptr<DataNode> const& entry, int indent) {
    if (auto p = std::dynamic_pointer_cast<data::DataArray>(entry)) {
        os << "[ ";
        auto it = entry->FirstChild();
        Print(os, it, indent + 1);
        it = it->Next();
        while (it != nullptr) {
            os << " , ";
            Print(os, it, indent);
        }
        os << " ]";
    } else if (auto p = std::dynamic_pointer_cast<data::DataTable>(entry)) {
        os << "{ ";
        auto it = entry->FirstChild();
        os << std::endl << std::setw(indent) << "\"" << it->Key() << "\" = ";
        Print(os, it->Value(), indent + 1);
        while (it != nullptr) {
            os << "," << std::endl << std::setw(indent) << "\"" << it->Key() << "\" = ";
            Print(os, it->Value(), indent + 1);
        }
        if (p->Count() > 1) { os << std::endl << std::setw(indent) << " "; }
        os << "}";
    } else {
        Print(os, entry->Value(), indent);
    }
    return os;
}
std::ostream& operator<<(std::ostream const& os, DataNode const& entry) {
    return Print(os, entry.shared_from_this(), 0);
}
}
}