//
// Created by salmon on 17-8-16.
//
#include "DataBaseStdIO.h"
#include "../DataArray.h"
#include "../DataBlock.h"
#include "../DataEntity.h"
#include "../DataTable.h"
#include "simpla/utilities/FancyStream.h"
namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseStdIO, stdio);

struct DataBaseStdIO::pimpl_s {
    std::ostream* m_out_ = nullptr;
    std::istream* m_in_ = nullptr;
};
DataBaseStdIO::DataBaseStdIO() : m_pimpl_(new pimpl_s) {}
DataBaseStdIO::~DataBaseStdIO() { delete m_pimpl_; }
void DataBaseStdIO::SetStream(std::ostream& out) { m_pimpl_->m_out_ = &out; }
void DataBaseStdIO::SetStream(std::istream& in) { m_pimpl_->m_in_ = &in; }
int DataBaseStdIO::Connect(std::string const& authority, std::string const& path, std::string const& query,
                           std::string const& fragment) {
    m_pimpl_->m_out_ = &(std::cout);
    m_pimpl_->m_in_ = &(std::cin);
    return SP_SUCCESS;
}

int DataBaseStdIO::Disconnect() {
    m_pimpl_->m_out_ = nullptr;
    m_pimpl_->m_in_ = nullptr;
    return SP_SUCCESS;
}

int DataBaseStdIO::Flush() {
    *m_pimpl_->m_out_ << std::endl;
    return SP_SUCCESS;
}
bool DataBaseStdIO::isNull() const { return m_pimpl_->m_out_ == nullptr; }

std::shared_ptr<DataEntity> DataBaseStdIO::Get(std::string const& URI) const { return nullptr; }

std::ostream& Print(std::ostream& os, std::shared_ptr<const DataEntity> const& v, int indent);

std::ostream& Print(std::ostream& os, std::shared_ptr<const DataLight> const& v, int indent) {
    if (auto p = std::dynamic_pointer_cast<const DataLightT<std::string>>(v)) {
        os << "\"" << p->value() << "\"";
    } else if (auto p = std::dynamic_pointer_cast<const DataLightT<bool>>(v)) {
        os << std::boolalpha << p->value();
    }

#define SP_TYPE_DISPATCH(_T_)                                                \
    else if (auto p = std::dynamic_pointer_cast<const DataLightT<_T_>>(v)) { \
        os << p->value();                                                    \
    }
    SP_TYPE_DISPATCH(int)
    SP_TYPE_DISPATCH(long)
    SP_TYPE_DISPATCH(short)
    SP_TYPE_DISPATCH(long long)
    SP_TYPE_DISPATCH(unsigned int)
    SP_TYPE_DISPATCH(unsigned long)
    SP_TYPE_DISPATCH(unsigned long long)
    SP_TYPE_DISPATCH(float)
    SP_TYPE_DISPATCH(double)
    SP_TYPE_DISPATCH(long double)
#undef SP_TYPE_DISPATCH

    else if (auto p = std::dynamic_pointer_cast<const DataLight>(v)) {
        os << "<" << p->value_type_info().name() << ">";
    }

    return os;
}

std::ostream& Print(std::ostream& os, std::shared_ptr<const DataBlock> const& p, int indent) {
    int ndims = p->GetNDIMS();
    std::vector<index_type> lo(ndims), hi(ndims);
    p->GetIndexBox(&lo[0], &hi[0]);
    os << "<Block[" << lo << "," << hi << "]>";
    return os;
}
std::ostream& Print(std::ostream& os, std::shared_ptr<const DataArray> const& p, int indent) {
    os << "[ ";
    Print(os, p->Get(0), indent + 1);
    for (size_type i = 1, n = p->Count(); i < n; ++i) {
        os << " , ";
        Print(os, p->Get(i), indent + 1);
    }
    os << " ]";
    return os;
}
std::ostream& Print(std::ostream& os, std::shared_ptr<const DataTable> const& p, int indent) {
    //    os << "<Table[" << p->Count() << "]>";
    os << "{";

    int count = 0;
    p->Foreach([&](std::string const& key, std::shared_ptr<DataEntity> v) {
        if (count > 0) { os << std::endl << std::setw(indent) << " , "; }
        os << "\"" << key << "\" = ";
        Print(os, v, indent + 1);
        ++count;
        return 1;
    });

    os << "}";

    return os;
}
std::ostream& Print(std::ostream& os, std::shared_ptr<const DataEntity> const& v, int indent) {
    if (auto p = std::dynamic_pointer_cast<const DataLight>(v)) {
        Print(os, p, indent);
    } else if (auto p = std::dynamic_pointer_cast<const DataArray>(v)) {
        Print(os, p, indent);
    } else if (auto p = std::dynamic_pointer_cast<const DataTable>(v)) {
        Print(os, p, indent);
    } else if (auto p = std::dynamic_pointer_cast<const DataBlock>(v)) {
        Print(os, p, indent);
    } else {
        os << "< illegal type >";
    }
    return os;
};

int DataBaseStdIO::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    *m_pimpl_->m_out_ << uri << "=";
    Print(*m_pimpl_->m_out_, v, 0);
    return 0;
}
int DataBaseStdIO::Add(std::string const& URI, const std::shared_ptr<DataEntity>& v) { return 0; }
int DataBaseStdIO::Delete(std::string const& URI) { return 0; }
int DataBaseStdIO::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return 0;
}
}
}