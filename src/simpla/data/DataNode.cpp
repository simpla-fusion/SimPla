//
// Created by salmon on 17-8-18.
//

#include "DataNode.h"
#include <iomanip>
#include "DataEntity.h"

#include "DataArray.h"
#include "DataTable.h"

namespace simpla {
namespace data {

void Print(std::ostream& os, std::shared_ptr<DataEntity> const&, int indent) {}

void Print(std::ostream& os, std::shared_ptr<DataNode> const& entry, int indent) {
    if (auto p = std::dynamic_pointer_cast<data::DataArray>(entry)) {
        os << "[ ";
        auto it = p->Child();
        Print(os, it, indent + 1);
        it = it->Next();
        while (it != nullptr) {
            os << " , ";
            Print(os, it, indent);
        }
        os << " ]";
    } else if (auto p = std::dynamic_pointer_cast<data::DataTable>(entry)) {
        os << "{ ";
        auto it = p->Child();
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
}
std::ostream& operator<<(std::ostream const& os, DataNode const& entry) {
    Print(os, entry.shared_from_this(), 0);
    return os;
}
}
}