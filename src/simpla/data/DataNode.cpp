//
// Created by salmon on 17-8-18.
//

#include "DataNode.h"
#include <iomanip>
#include <map>
#include <vector>
#include "DataBase.h"
#include "DataEntity.h"
namespace simpla {
namespace data {

DataNode::DataNode() = default;
DataNode::~DataNode() = default;
std::shared_ptr<DataNode> DataNode::New(std::string const& s) {
    return s.empty() ? std::shared_ptr<DataNode>(new DataNode) : data::DataBase::New(s)->Root();
}

std::ostream& Print(std::ostream& os, std::shared_ptr<const DataNode> const& entry, int indent) {
    if (entry->isArray()) {
        os << "[ ";
        auto it = entry->FirstChild();
        Print(os, it, indent + 1);
        it = it->Next();
        while (it != nullptr) {
            os << " , ";
            Print(os, it, indent);
        }
        os << " ]";
    } else if (entry->isTable()) {
        os << "{ ";
        auto it = entry->FirstChild();
        if (it != nullptr) {
            os << std::endl << std::setw(indent) << "\"" << it->GetKey() << "\" = ";
            Print(os, it, indent + 1);
            it = it->Next();
            while (it != nullptr) {
                os << "," << std::endl << std::setw(indent) << "\"" << it->GetKey() << "\" = ";
                Print(os, it, indent + 1);
            }
            if (entry->GetNumberOfChildren() > 1) { os << std::endl << std::setw(indent) << " "; }
            os << "}";
        }
    } else if (entry->isEntity()) {
        os << *entry->GetValue();
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, DataNode const& entry) { return Print(os, entry.shared_from_this(), 0); }
}  // namespace data {
}  // namespace simpla {
