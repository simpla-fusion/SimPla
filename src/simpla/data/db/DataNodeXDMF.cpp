//
// Created by salmon on 17-8-13.
//
#include <sys/stat.h>
#include <fstream>

#include <Xdmf.hpp>
#include <XdmfAttributeCenter.hpp>
#include <XdmfDomain.hpp>
#include <XdmfGridCollection.hpp>
#include <XdmfWriter.hpp>

#include <simpla/parallel/MPIComm.h>

#include "../DataNode.h"
#include "DataNodeMemory.h"

namespace simpla {
namespace data {
int XDMFDump(std::string url, std::shared_ptr<DataNode> const& v);

struct DataNodeXDMF : public DataNodeMemory {
    SP_DATA_NODE_HEAD(DataNodeXDMF, DataNodeMemory)

    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override;
    int Disconnect() override { return Flush(); }
    bool isValid() const override { return true; }
    int Flush() override { return XDMFDump(m_file_, shared_from_this()); }

   private:
    std::string m_file_;
};
REGISTER_CREATOR(DataNodeXDMF, xmf);
DataNodeXDMF::DataNodeXDMF(DataNode::eNodeType etype) : base_type(etype) {}
DataNodeXDMF::~DataNodeXDMF() = default;

int DataNodeXDMF::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    m_file_ = path;

#ifdef MPI_FOUND
    if (GLOBAL_COMM.size() > 1) {
        auto pos = path.rfind('.');
        auto prefix = (pos != std::string::npos) ? path.substr(0, pos) : path;
        auto ext = (pos != std::string::npos) ? path.substr(pos + 1) : path;
        int digital = static_cast<int>(std::floor(std::log(static_cast<double>(GLOBAL_COMM.size())))) + 1;
        if (GLOBAL_COMM.rank() == 0) {
            std::ofstream summary(prefix + ".summary.xmf");
            summary << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
            summary << "<Xdmf Version=\"2.0\" xmlns:xi=\"http://www.w3.org/2001/XInclude\">" << std::endl;
            for (int i = 0, ie = GLOBAL_COMM.size(); i < ie; ++i) {
                summary << "  <xi:include href=\"" << prefix << "." << std::setfill('0') << std::setw(digital) << i
                        << ".xmf\"/>" << std::endl;
            }
            summary << "</Xdmf>";
        }
        std::ostringstream os;
        os << prefix << "." << std::setfill('0') << std::setw(digital) << GLOBAL_COMM.rank() << ".xmf";
        m_file_ = os.str();
    }
#endif
    return SP_SUCCESS;
}
template <typename U>
struct XdmfType {};
template <>
struct XdmfType<double> {
    static auto type() { return XdmfArrayType::Float64(); }
};

template <>
struct XdmfType<float> {
    static auto type() { return XdmfArrayType::Float32(); }
};
template <>
struct XdmfType<int> {
    static auto type() { return XdmfArrayType::Int32(); }
};
template <>
struct XdmfType<long> {
    static auto type() { return XdmfArrayType::Int64(); }
};
template <>
struct XdmfType<unsigned int> {
    static auto type() { return XdmfArrayType::UInt32(); }
};
int XDMFWriteArray(XdmfArray* dst, std::shared_ptr<DataNode> const& data) {
    if (data == nullptr) { return 0; }
    if (data->type() == DataNode::DN_TABLE) {
    } else if (data->type() == DataNode::DN_ARRAY) {
        unsigned int dof = data->size();
        nTuple<index_type, 3> lo, hi, dims;
        if (auto blk = std::dynamic_pointer_cast<DataBlock>(data->Get(0)->GetEntity())) {
            blk->GetIndexBox(&lo[0], &hi[0]);
        } else {
            RUNTIME_ERROR << *data;
        }

        std::vector<unsigned int> dimensions{static_cast<unsigned int>(hi[0] - lo[0]),
                                             static_cast<unsigned int>(hi[1] - lo[1]),
                                             static_cast<unsigned int>(hi[2] - lo[2]), dof};
        for (int i = 0; i < dof; ++i) {
            if (auto block = std::dynamic_pointer_cast<DataBlockT<double>>(data->Get(i)->GetEntity())) {
                dst->initialize(XdmfType<double>::type(), dimensions);
                dst->insert(i, block->get(), block->size(), dof, 1);
            } else if (auto block = std::dynamic_pointer_cast<DataBlockT<float>>(data->Get(i)->GetEntity())) {
                dst->initialize(XdmfType<float>::type(), dimensions);
                dst->insert(i, block->get(), block->size(), dof, 1);
            } else if (auto block = std::dynamic_pointer_cast<DataBlockT<int>>(data->Get(i)->GetEntity())) {
                dst->initialize(XdmfType<int>::type(), dimensions);
                dst->insert(i, block->get(), block->size(), dof, 1);
            } else if (auto block = std::dynamic_pointer_cast<DataBlockT<long>>(data->Get(i)->GetEntity())) {
                dst->initialize(XdmfType<long>::type(), dimensions);
                dst->insert(i, block->get(), block->size(), dof, 1);
            } else if (auto block = std::dynamic_pointer_cast<DataBlockT<unsigned int>>(data->Get(i)->GetEntity())) {
                dst->initialize(XdmfType<unsigned int>::type(), dimensions);
                dst->insert(i, block->get(), block->size(), dof, 1);
            } else {
                UNIMPLEMENTED;
            }
        }
    } else if (auto blk = std::dynamic_pointer_cast<DataBlock>(data->GetEntity())) {
        nTuple<index_type, 3> lo, hi;
        blk->GetIndexBox(&lo[0], &hi[0]);
        std::vector<unsigned int> dimensions{static_cast<unsigned int>(hi[0] - lo[0]),
                                             static_cast<unsigned int>(hi[1] - lo[1]),
                                             static_cast<unsigned int>(hi[2] - lo[2])};

        if (auto block = std::dynamic_pointer_cast<DataBlockT<double>>(blk)) {
            dst->initialize(XdmfType<double>::type(), dimensions);
            dst->insert(0, block->get(), block->size(), 1, 1);
        } else if (auto block = std::dynamic_pointer_cast<DataBlockT<float>>(blk)) {
            dst->initialize(XdmfType<float>::type(), dimensions);
            dst->insert(0, block->get(), block->size(), 1, 1);
        } else if (auto block = std::dynamic_pointer_cast<DataBlockT<int>>(blk)) {
            dst->initialize(XdmfType<int>::type(), dimensions);
            dst->insert(0, block->get(), block->size(), 1, 1);
        } else if (auto block = std::dynamic_pointer_cast<DataBlockT<long>>(blk)) {
            dst->initialize(XdmfType<long>::type(), dimensions);
            dst->insert(0, block->get(), block->size(), 1, 1);
        } else if (auto block = std::dynamic_pointer_cast<DataBlockT<unsigned int>>(blk)) {
            dst->initialize(XdmfType<unsigned int>::type(), dimensions);
            dst->insert(0, block->get(), block->size(), 1, 1);
        } else {
            UNIMPLEMENTED;
        }
    }

    //            (auto p = std::dynamic_pointer_cast<DataBlockT<double>>(data->GetEntity())) {
    //        XDMFWriteArrayT(dst, p);
    //    } else if (auto p = std::dynamic_pointer_cast<DataBlockT<float>>(data->GetEntity())) {
    //        XDMFWriteArrayT(dst, p);
    //    } else if (auto p = std::dynamic_pointer_cast<DataBlockT<int>>(data->GetEntity())) {
    //        XDMFWriteArrayT(dst, p);
    //    } else if (auto p = std::dynamic_pointer_cast<DataBlockT<long>>(data->GetEntity())) {
    //        XDMFWriteArrayT(dst, p);
    //    } else if (auto p = std::dynamic_pointer_cast<DataBlockT<unsigned int>>(data->GetEntity())) {
    //        XDMFWriteArrayT(dst, p);
    //    }
    return 1;
}
template <typename T>
size_type XDMFAttributeInsertOne(T& grid, std::string const& s_name, std::shared_ptr<data::DataNode> const& node) {
    size_type count = 0;
    auto attr = XdmfAttribute::New();
    attr->setName(s_name);
    auto iform = node->GetValue<int>("IFORM");
    switch (iform) {
        case NODE:
            attr->setCenter(XdmfAttributeCenter::Node());
            break;
        case CELL:
            attr->setCenter(XdmfAttributeCenter::Cell());
            break;
        case EDGE:
            attr->setCenter(XdmfAttributeCenter::Edge());
            break;
        case FACE:
            attr->setCenter(XdmfAttributeCenter::Face());
            break;
        default:
            UNIMPLEMENTED;
            break;
    }
    size_type rank = 0;
    size_type extents[MAX_NDIMS_OF_ARRAY];
    if (auto p = node->Get("DOF")) {
        if (auto dof = std::dynamic_pointer_cast<DataLightT<int>>(p->GetEntity())) {
            switch (dof->value()) {
                case 1:
                    attr->setType(XdmfAttributeType::Scalar());
                    break;
                case 3:
                    attr->setType(XdmfAttributeType::Vector());
                    break;
                case 6:
                    attr->setType(XdmfAttributeType::Tensor6());
                    break;
                case 9:
                    attr->setType(XdmfAttributeType::Tensor());
                    break;
                default:
                    attr->setType(XdmfAttributeType::Matrix());
                    break;
            }
            rank = dof->value() > 1 ? 1 : 0;
            extents[0] = static_cast<size_type>(dof->value());
        } else if (auto dof = std::dynamic_pointer_cast<DataLightT<int*>>(p->GetEntity())) {
            attr->setType(XdmfAttributeType::Matrix());
            rank = dof->extents(extents);
        }
    } else {
        attr->setType(XdmfAttributeType::Scalar());
    }

    if (iform == NODE || iform == CELL) {
        //        auto v_type = node->GetValue<std::string>("ValueType", "double");
        //
        //        if (v_type == "double") {
        //            XDMFWriteArray<double>(attr.get(), node->Get("_DATA_"));
        //        } else if (v_type == "float") {
        //            XDMFWriteArray<float>(attr.get(), node->Get("_DATA_"));
        //        } else if (v_type == "int") {
        //            XDMFWriteArray<int>(attr.get(), node->Get("_DATA_"));
        //        } else if (v_type == "long") {
        //            XDMFWriteArray<long>(attr.get(), node->Get("_DATA_"));
        //        }
        XDMFWriteArray(attr.get(), node->Get("_DATA_"));
        grid->insert(attr);
        count = 1;

    } else {
        VERBOSE << std::setw(20) << "Write XDMF : "
                << "Ignore EDGE/FACE center attribute \"" << s_name << "\".";
    }
    return count;
}
template <typename T>
size_type XDMFAttributeInsert(T& grid, std::shared_ptr<data::DataNode> const& attrs) {
    return attrs->Foreach([&](std::string const& k, std::shared_ptr<data::DataNode> const& node) {
        return XDMFAttributeInsertOne(grid, k, node);
    });
}
boost::shared_ptr<XdmfCurvilinearGrid> XDMFCurvilinearGridNew(std::shared_ptr<DataNode> const& chart,
                                                              std::shared_ptr<data::DataNode> const& patch) {
    auto coord = patch->Get("Attributes/_COORDINATES_/_DATA_");

    ASSERT(coord != nullptr && coord->type() == DataNode::DN_ARRAY);
    auto x = std::dynamic_pointer_cast<DataBlockT<Real>>(coord->Get(0)->GetEntity());
    auto y = std::dynamic_pointer_cast<DataBlockT<Real>>(coord->Get(1)->GetEntity());
    auto z = std::dynamic_pointer_cast<DataBlockT<Real>>(coord->Get(2)->GetEntity());
    nTuple<index_type, 3> lo{0, 0, 0}, hi{0, 0, 0};
    x->GetIndexBox(&lo[0], &hi[0]);
    nTuple<unsigned int, 3> dims{0, 0, 0};
    dims = hi - lo;
    unsigned int num = dims[0] * dims[1] * dims[2];
    auto grid = XdmfCurvilinearGrid::New(dims[0], dims[1], dims[2]);
    auto geo = XdmfGeometry::New();
    geo->setType(XdmfGeometryType::XYZ());
    nTuple<Real, 3> origin = chart->GetValue<nTuple<Real, 3>>("Origin");
    origin += lo * chart->GetValue<nTuple<Real, 3>>("Scale");
    geo->setOrigin(origin[0], origin[1], origin[2]);

    XDMFWriteArray(geo.get(), coord);

    grid->setGeometry(geo);

    XDMFAttributeInsert(grid, patch->Get("Attributes"));
    return grid;
}
boost::shared_ptr<XdmfRegularGrid> XDMFRegularGridNew(std::shared_ptr<DataNode> const& chart,
                                                      std::shared_ptr<data::DataNode> const& patch) {
    auto x0 = chart->GetValue<nTuple<Real, 3>>("Origin");
    auto dx = chart->GetValue<nTuple<Real, 3>>("Scale");
    auto idx_box = patch->GetValue<index_box_type>("Block");
    nTuple<unsigned int, 3> dims;
    dims = std::get<1>(idx_box) - std::get<0>(idx_box);
    nTuple<unsigned int, 3> origin;
    origin = std::get<0>(idx_box) * dx + x0;
    auto grid = XdmfRegularGrid::New(dx[0], dx[1], dx[2], dims[0], dims[1], dims[2], origin[0], origin[1], origin[2]);
    XDMFAttributeInsert(grid, patch->Get("Attributes"));
    return grid;
}
int XDMFDump(std::string url, std::shared_ptr<DataNode> const& obj) {
    if (obj == nullptr || url.empty()) { return SP_FAILED; }
    VERBOSE << std::setw(20) << "Write XDMF : " << url;

    int success = SP_FAILED;
    auto grid_collection = XdmfGridCollection::New();
    grid_collection->setType(XdmfGridCollectionType::Spatial());

    if (auto time = obj->Get("Time")) {
        if (auto t = std::dynamic_pointer_cast<DataLightT<Real>>(time->GetEntity())) {
            grid_collection->setTime(XdmfTime::New(t->value()));
        }
    }
    if (auto chart = obj->Get("Chart"))
        if (auto patch = obj->Get("Patch")) {
            patch->Foreach([&](std::string const& k, std::shared_ptr<data::DataNode> const& node) {
                if (node->Get("Attributes/_COORDINATES_") != nullptr) {
                    auto g = XDMFCurvilinearGridNew(chart, node);
                    g->setName(k);
                    grid_collection->insert(g);
                } else {
                    auto g = XDMFRegularGridNew(chart, node);
                    g->setName(k);
                    grid_collection->insert(g);
                }
                return 1;
            });
        }

    auto domain = XdmfDomain::New();
    domain->insert(grid_collection);
    if (auto writer = XdmfWriter::New(url)) {
        domain->accept(writer);
        success = SP_SUCCESS;
    }
    return success;
}

}  // namespace data{
}  // namespace simpla