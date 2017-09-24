//
// Created by salmon on 17-8-13.
//
#include <sys/stat.h>
#include <fstream>

//#include <Xdmf.hpp>
//#include <XdmfAttributeCenter.hpp>
//#include <XdmfDomain.hpp>
//#include <XdmfGridCollection.hpp>
//#include <XdmfWriter.hpp>
//#include <XdmfHDF5Writer.hpp>
//#include <XdmfInformation.hpp>

#include <simpla/parallel/MPIComm.h>
#include "../DataNode.h"
#include "HDF5Common.h"

#include "DataNodeMemory.h"

namespace simpla {
namespace data {

struct DataNodeXDMF : public DataNodeMemory {
    SP_DATA_NODE_HEAD(DataNodeXDMF, DataNodeMemory)

    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override;
    int Disconnect() override;
    int Flush() override;
    bool isValid() const override { return !m_prefix_.empty(); }

    void WriteDataItem(std::string const& url, std::string const& key, std::shared_ptr<data::DataNode> const& array,
                       int indent = 0);
    void WriteAttribute(std::string const& url, std::shared_ptr<data::DataNode> const& attr_desc,
                        std::shared_ptr<data::DataNode> const& data, int indent = 0);

    std::string m_prefix_;
    std::string m_h5_prefix_;
    std::string m_ext_;
    std::ostringstream os;
    hid_t m_h5_file_;
    hid_t m_h5_root_;
};
REGISTER_CREATOR(DataNodeXDMF, xdmf);
DataNodeXDMF::DataNodeXDMF(DataNode::eNodeType etype) : base_type(etype) {}
DataNodeXDMF::~DataNodeXDMF() = default;

int DataNodeXDMF::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    m_prefix_ = path;

    auto pos = path.rfind('.');
    m_prefix_ = (pos != std::string::npos) ? path.substr(0, pos) : path;
    m_ext_ = (pos != std::string::npos) ? path.substr(pos + 1) : path;

    //    domain->insert(grid_collection);
    m_h5_prefix_ = m_prefix_;

#ifdef MPI_FOUND
    if (GLOBAL_COMM.size() > 1) {
        std::ostringstream os;
        int digital = static_cast<int>(std::floor(std::log(static_cast<double>(GLOBAL_COMM.size())))) + 1;
        os << m_h5_prefix_ << "." << std::setfill('0') << std::setw(digital) << GLOBAL_COMM.rank();
        m_h5_prefix_ = os.str();
    }
#endif
    m_h5_prefix_ = m_h5_prefix_ + ".h5";
    m_h5_file_ = H5Fcreate(m_h5_prefix_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    m_h5_root_ = H5Gopen(m_h5_file_, "/", H5P_DEFAULT);
    return SP_SUCCESS;
}

int DataNodeXDMF::Disconnect() {
    Flush();
    H5Gclose(m_h5_root_);
    H5Fclose(m_h5_file_);
    return SP_SUCCESS;
}

std::string XDMFNumberType(std::type_info const& t_info) {
    std::string number_type;
    if (t_info == typeid(double)) {
        number_type = R"(NumberType="Float" Precision="8")";
    } else if (t_info == typeid(float)) {
        number_type = R"(NumberType="Float" Precision="4" )";
    } else if (t_info == typeid(int)) {
        number_type = R"(NumberType="Int" )";
    } else if (t_info == typeid(unsigned int)) {
        number_type = R"(NumberType="UInt" )";
    } else {
    }
    return number_type;
}

void DataNodeXDMF::WriteDataItem(std::string const& url, std::string const& key,
                                 std::shared_ptr<data::DataNode> const& data, int indent) {
    int ndims = 0;
    int dof = 0;
    std::string number_type;
    index_type hi[MAX_NDIMS_OF_ARRAY];
    index_type lo[MAX_NDIMS_OF_ARRAY];
    for (auto& v : lo) { v = std::numeric_limits<index_type>::max(); }
    for (auto& v : hi) { v = std::numeric_limits<index_type>::min(); }

    auto g_id = H5GroupTryOpen(m_h5_root_, url);
    //    bool is_exist = H5Lexists(g_id, key.c_str(), H5P_DEFAULT) != 0;
    //    H5O_info_t g_info;
    //    if (is_exist) { H5_ERROR(H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT)); }
    //
    //    if (is_exist && g_info.type != H5O_TYPE_DATASET) {
    //        H5Ldelete(g_id, key.c_str(), H5P_DEFAULT);
    //        is_exist = false;
    //    }

    if (auto array = std::dynamic_pointer_cast<ArrayBase>(data->GetEntity())) {
        number_type = XDMFNumberType(array->value_type_info());
        ndims = array->GetIndexBox(lo, hi);

        index_type inner_lower[MAX_NDIMS_OF_ARRAY];
        index_type inner_upper[MAX_NDIMS_OF_ARRAY];
        index_type outer_lower[MAX_NDIMS_OF_ARRAY];
        index_type outer_upper[MAX_NDIMS_OF_ARRAY];

        array->GetIndexBox(inner_lower, inner_upper);
        array->GetIndexBox(outer_lower, outer_upper);

        hsize_t m_shape[MAX_NDIMS_OF_ARRAY];
        hsize_t m_start[MAX_NDIMS_OF_ARRAY];
        hsize_t m_count[MAX_NDIMS_OF_ARRAY];
        hsize_t m_stride[MAX_NDIMS_OF_ARRAY];
        hsize_t m_block[MAX_NDIMS_OF_ARRAY];
        for (int i = 0; i < ndims; ++i) {
            m_shape[i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
            m_start[i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
            m_count[i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
            m_stride[i] = static_cast<hsize_t>(1);
            m_block[i] = static_cast<hsize_t>(1);
        }
        hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
        H5_ERROR(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], &m_stride[0], &m_count[0], &m_block[0]));
        hid_t f_space = H5Screate_simple(ndims, &m_count[0], nullptr);
        hid_t dset;
        hid_t d_type = H5NumberType(array->value_type_info());
        H5_ERROR(dset = H5Dcreate(g_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, array->pointer()));

        H5_ERROR(H5Dclose(dset));
        if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
        if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
    } else if (data->type() == data::DataNode::DN_ARRAY) {
        dof = data->size();
        hid_t d_type = H5T_NO_CLASS;

        for (int i = 0; i < dof; ++i) {
            if (auto array = std::dynamic_pointer_cast<ArrayBase>(data->GetEntity(i))) {
                if (d_type == H5T_NO_CLASS) { d_type = H5NumberType(array->value_type_info()); }
                auto t_ndims = array->GetNDIMS();
                index_type t_lo[MAX_NDIMS_OF_ARRAY], t_hi[MAX_NDIMS_OF_ARRAY];
                array->GetIndexBox(t_lo, t_hi);
                ndims = std::max(ndims, array->GetNDIMS());
                for (int n = 0; n < t_ndims; ++n) {
                    lo[n] = std::min(lo[n], t_lo[n]);
                    hi[n] = std::max(hi[n], t_hi[n]);
                }
            }
        }

        hsize_t f_shape[MAX_NDIMS_OF_ARRAY];

        for (int i = 0; i < ndims; ++i) { f_shape[i] = static_cast<hsize_t>(hi[i] - lo[i]); }
        f_shape[ndims] = dof;

        hid_t dset;
        hid_t f_space = H5Screate_simple(ndims + 1, &f_shape[0], nullptr);
        hid_t plist = H5P_DEFAULT;
        if (H5Tequal(d_type, H5T_NATIVE_DOUBLE)) {
            plist = H5Pcreate(H5P_DATASET_CREATE);
            double fillval = std::numeric_limits<double>::quiet_NaN();
            H5_ERROR(H5Pset_fill_value(plist, H5T_NATIVE_DOUBLE, &fillval));
        }
        H5_ERROR(dset = H5Dcreate(g_id, key.c_str(), d_type, f_space, H5P_DEFAULT, plist, H5P_DEFAULT));
        H5_ERROR(H5Pclose(plist));
        H5_ERROR(H5Sclose(f_space));

        for (int i = 0; i < dof; ++i) {
            if (auto array = std::dynamic_pointer_cast<ArrayBase>(data->GetEntity(i))) {
                auto t_ndims = array->GetNDIMS();
                index_type t_lo[t_ndims], t_hi[t_ndims];
                array->GetIndexBox(t_lo, t_hi);

                hsize_t m_shape[MAX_NDIMS_OF_ARRAY];

                hsize_t f_start[MAX_NDIMS_OF_ARRAY];
                hsize_t f_count[MAX_NDIMS_OF_ARRAY];
                hsize_t f_stride[MAX_NDIMS_OF_ARRAY];
                hsize_t f_block[MAX_NDIMS_OF_ARRAY];
                for (int i = 0; i < t_ndims; ++i) {
                    m_shape[i] = static_cast<hsize_t>(t_hi[i] - t_lo[i]);
                    f_start[i] = static_cast<hsize_t>(t_lo[i] - lo[i]);
                    f_count[i] = static_cast<hsize_t>(t_hi[i] - t_lo[i]);
                    f_stride[i] = static_cast<hsize_t>(1);
                    f_block[i] = static_cast<hsize_t>(1);
                }
                for (int i = t_ndims; i < ndims; ++i) {
                    m_shape[i] = static_cast<hsize_t>(1);
                    f_start[i] = static_cast<hsize_t>(0);
                    f_count[i] = static_cast<hsize_t>(1);
                    f_stride[i] = static_cast<hsize_t>(1);
                    f_block[i] = static_cast<hsize_t>(1);
                }
                f_start[ndims] = i;
                f_stride[ndims] = 1;
                f_count[ndims] = 1;
                f_block[ndims] = 1;

                hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
                f_space = H5Screate_simple(ndims + 1, &f_shape[0], nullptr);
                H5_ERROR(
                    H5Sselect_hyperslab(f_space, H5S_SELECT_SET, &f_start[0], &f_stride[0], &f_count[0], &f_block[0]));
                H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, array->pointer()));

                if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
                if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
            }
        }
        H5_ERROR(H5Dclose(dset));

    } else {
        UNIMPLEMENTED;
    }

    H5Gclose(g_id);

    os << std::setw(indent) << " "
       << "<DataItem Format=\"HDF\" " << number_type << " Dimensions=\"";
    os << hi[0] - lo[0];
    for (int i = 1; i < ndims; ++i) { os << " " << hi[i] - lo[i]; };
    if (dof > 0) { os << " " << dof; }
    os << "\">" << m_h5_prefix_ << ":" << url << "/" << key << "</DataItem>" << std::endl;
}
void DataNodeXDMF::WriteAttribute(std::string const& url, std::shared_ptr<data::DataNode> const& attr_desc,
                                  std::shared_ptr<data::DataNode> const& data, int indent) {
    static const char* attr_center[] = {"Node", "Node" /* "Edge"*/, "Node" /* "Face"*/, "Cell", "Grid", "Other"};
    //    static const char* attr_type[] = {" Scalar", "Vector", "Tensor", "Tensor6", "Matrix", "GlobalID"};

    std::string attr_type = "Scalar";
    size_type rank = 0;
    size_type extents[MAX_NDIMS_OF_ARRAY] = {1, 1, 1, 1};
    if (auto p = attr_desc->Get("DOF")) {
        if (auto dof_t = std::dynamic_pointer_cast<DataLightT<int>>(p->GetEntity())) {
            auto dof = static_cast<size_type>(dof_t->value());
            switch (dof_t->value()) {
                case 1:
                    attr_type = "Scalar";
                    break;
                case 3:
                    attr_type = "Vector";
                    break;
                default:
                    attr_type = "Matrix";
                    break;
            }
            rank = dof > 1 ? 1 : 0;
            extents[0] = dof;
        } else if (auto dof_p = std::dynamic_pointer_cast<DataLightT<int*>>(p->GetEntity())) {
            attr_type = "Matrix";
            rank = dof_p->extents(extents);
        }
    }
    std::string s_name = attr_desc->GetValue<std::string>("Name");
    os << std::setw(indent) << " "
       << "<Attribute "
       << "Center=\"" << attr_center[attr_desc->GetValue<int>("IFORM", 0)] << "\" "  //
       << "Name=\"" << s_name << "\" "                                               //
       << "AttributeType=\"" << attr_type << "\" "                                   //
       << "IFORM=\"" << attr_desc->GetValue<int>("IFORM", 0) << "\" "                //
       << ">" << std::endl;

    WriteDataItem(url, s_name, data, indent + 1);

    os << std::setw(indent) << " "
       << "</Attribute>" << std::endl;
}

void XDMFGeometryCurvilinear(DataNodeXDMF* self, std::string const& prefix, std::shared_ptr<DataNode> const& chart,
                             std::shared_ptr<data::DataNode> const& blk, std::shared_ptr<data::DataNode> const& coord,
                             int indent) {
    auto lo = blk->GetValue<nTuple<index_type, 3>>("LowIndex");
    auto hi = blk->GetValue<nTuple<index_type, 3>>("HighIndex");

    nTuple<unsigned int, 3> dims{0, 0, 0};
    dims = hi - lo + 1;

    self->os << std::setw(indent) << " "
             << R"(<Topology TopologyType="3DSMesh" Dimensions=")" << dims[0] << " " << dims[1] << " " << dims[2]
             << "\" />" << std::endl;
    self->os << std::setw(indent) << " "
             << "<Geometry GeometryType=\"XYZ\">" << std::endl;
    self->WriteDataItem(prefix, "_COORDINATES_", coord, indent + 1);
    self->os << std::setw(indent) << " "
             << "</Geometry>" << std::endl;
}
void XDMFGeometryRegular(DataNodeXDMF* self, std::shared_ptr<DataNode> const& chart,
                         std::shared_ptr<data::DataNode> const& blk, int indent = 0) {
    auto x0 = chart->GetValue<nTuple<Real, 3>>("Origin");
    auto dx = chart->GetValue<nTuple<Real, 3>>("Scale");
    auto lo = blk->GetValue<index_tuple>("LowIndex");
    auto hi = blk->GetValue<index_tuple>("HighIndex");

    //    nTuple<unsigned int, 3> dims{1, 1, 1};
    //    dims = hi - lo + 1;
    //    nTuple<Real, 3> origin{0, 0, 0};
    //    origin = lo * dx + x0;
    //    auto grid = XdmfRegularGrid::New(dx[0], dx[1], dx[2], dims[0], dims[1], dims[2], origin[0], origin[1],
    //    origin[2]);
    //    auto level_info = XdmfInformation::New();
    //    level_info->setKey("Level");
    //    level_info->setValue(std::to_string(blk->GetValue<int>("Level", 0)));
    //    grid->insert(level_info);
    //    grid->setName(std::to_string(blk->GetValue<id_type>("GUID")));
}
int DataNodeXDMF::Flush() {
    int success = SP_FAILED;

    auto attrs = this->Get("Attributes");

    auto patches = this->Get("Patches");

    ASSERT(attrs != nullptr && patches != nullptr);

    int indent = 2;
    if (auto atlas = this->Get("Atlas")) {
        auto chart = atlas->Get("Chart");
        auto blks = atlas->Get("Blocks");

        blks->Foreach([&](std::string const& k, std::shared_ptr<data::DataNode> const& blk) {
            auto guid = blk->GetValue<id_type>("GUID");
            os << std::setw(indent) << " "
               << "<Grid Name=\"" << guid << "\" Level=\"" << blk->GetValue<int>("Level", 0) << "\">" << std::endl;
            if (auto patch = patches->Get(k)) {
                if (patch->Get("_COORDINATES_") != nullptr) {
                    XDMFGeometryCurvilinear(this, "/Patches/" + std::to_string(guid), chart, blk,
                                            patch->Get("_COORDINATES_"), indent + 1);
                } else {
                    XDMFGeometryRegular(this, chart, blk, indent + 1);
                }

                patch->Foreach([&](std::string const& s, std::shared_ptr<data::DataNode> const& d) {
                    if (attrs->Get(s)->Check("COORDINATES")) { return; }
                    WriteAttribute("/Patches/" + std::to_string(guid), attrs->Get(s), d, indent + 1);
                });
                os << std::setw(indent) << " "
                   << "</Grid>" << std::endl;
            }
            return 1;
        });
    }

    // Gather String

    std::string grid_str = os.str();

    if (!grid_str.empty()) {
        auto begin = grid_str.find("<Grid ");
        auto end = grid_str.find("</Xdmf>");
        grid_str = grid_str.substr(begin, end - begin);
    }

#ifdef MPI_FOUND
    grid_str = parallel::gather_string(grid_str);
#endif  // MPI_FOUND

    if (GLOBAL_COMM.rank() == 0) {
        std::ofstream out_file;
        out_file.open(m_prefix_ + ".xdmf", std::ios_base::trunc);
        VERBOSE << std::setw(20) << "Write XDMF : " << m_prefix_ << ".xdmf";

        out_file << R"(<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.3">
<Domain>
<Grid CollectionType="Spatial" GridType="Collection" Name="Collection">)";
        if (auto time = this->Get("Time")) {
            if (auto t = std::dynamic_pointer_cast<DataLightT<Real>>(time->GetEntity())) {
                out_file << std::endl << "  <Time Value=\"" << t->value() << "\"/>" << std::endl;
            }
        }
        out_file << grid_str;
        out_file << R"(</Grid>
</Domain>
</Xdmf>)";

        out_file.close();
    }
    return success;
}

}  // namespace data{
}  // namespace simpla