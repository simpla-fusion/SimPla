//
// Created by salmon on 17-8-13.
//
#include <sys/stat.h>
#include <fstream>

#include <simpla/parallel/MPIComm.h>
#include "../DataEntry.h"
#include "HDF5Common.h"

#include "DataEntryMemory.h"

namespace simpla {
namespace data {

struct DataEntryXDMF : public DataEntryMemory {
    SP_DATA_ENTITY_HEAD(DataEntryMemory, DataEntryXDMF, xdmf)

    int Connect(std::string const &authority, std::string const &path, std::string const &query,
                std::string const &fragment) override;
    int Disconnect() override;
    int Flush() override;
    bool isValid() const override { return !m_prefix_.empty(); }

    void WriteDataItem(std::string const &url, std::string const &key, const index_box_type &idx_box,
                       std::shared_ptr<const data::DataEntry> const &data, int indent);
    void WriteAttribute(std::string const &url, std::string const &key, const index_box_type &idx_box,
                        std::shared_ptr<const data::DataEntry> const &attr, int indent);
    void WriteParticle(std::string const &url, std::string const &key, const index_box_type &idx_box,
                       std::shared_ptr<const data::DataEntry> const &data, int indent);

    std::string m_prefix_;
    std::string m_h5_prefix_;
    std::string m_ext_;
    std::shared_ptr<std::ostringstream> m_os_;
    hid_t m_h5_file_ = 0;
    hid_t m_h5_root_ = 0;
};
SP_REGISTER_CREATOR(DataEntry, DataEntryXDMF);

DataEntryXDMF::DataEntryXDMF(DataEntry::eNodeType etype)
    : DataEntryMemory(etype), m_os_(std::make_shared<std::ostringstream>()) {}
DataEntryXDMF::DataEntryXDMF(DataEntryXDMF const &other) = default;
DataEntryXDMF::~DataEntryXDMF() { Disconnect(); };

int DataEntryXDMF::Connect(std::string const &authority, std::string const &path, std::string const &query,
                           std::string const &fragment) {
    m_prefix_ = path;
    auto pos = path.rfind('.');
    m_prefix_ = (pos != std::string::npos) ? path.substr(0, pos) : path;
    m_ext_ = (pos != std::string::npos) ? path.substr(pos + 1) : path;
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

int DataEntryXDMF::Disconnect() {
    if (m_h5_root_ > 0) {
        H5Gclose(m_h5_root_);
        m_h5_root_ = -1;
    }
    if (m_h5_file_ > 0) {
        H5Fclose(m_h5_file_);
        m_h5_file_ = -1;
    }

    return SP_SUCCESS;
}

std::string XDMFNumberType(std::type_info const &t_info) {
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
std::ostream &XDMFWriteArray(std::ostream &os, hid_t g_id, std::string const &prefix, std::string const &key,
                             index_type const *lo, index_type const *hi, std::shared_ptr<const ArrayBase> const &array,
                             int indent) {
    HDF5WriteArray(g_id, key, array);
    auto ndims = array->GetNDIMS();
    index_type m_lo[ndims];
    index_type m_hi[ndims];
    array->GetIndexBox(m_lo, m_hi);
    if (lo != nullptr && hi != nullptr) {
        os << std::setw(indent) << " "
           << R"(<DataItem ItemType="HyperSlab" Type="HyperSlab" Dimensions=")";
        for (int i = 0; i < ndims; ++i) { os << " " << hi[i] - lo[i]; };
        os << R"(" >)" << std::endl;
        os << std::setw(indent + 1) << " "
           << R"(<DataItem Dimensions="3 )" << ndims << R"("  Format="XML">)" << std::endl;
        os << std::setw(indent + 2) << " ";
        for (int i = 0; i < ndims; ++i) { os << " " << lo[i] - m_lo[i]; };
        os << std::endl;
        os << std::setw(indent + 2) << " ";
        for (int i = 0; i < ndims; ++i) { os << " " << 1; };
        os << std::endl;
        os << std::setw(indent + 2) << " ";
        for (int i = 0; i < ndims; ++i) { os << " " << hi[i] - lo[i]; };
        os << std::endl;
        os << std::setw(indent + 1) << " "
           << R"(</DataItem>)" << std::endl;
    }
    os << std::setw(indent + 1) << " "
       << "<DataItem Format=\"HDF\" " << XDMFNumberType(array->value_type_info()) << " Dimensions=\"";
    for (int i = 0; i < ndims; ++i) { os << " " << m_hi[i] - m_lo[i]; };
    os << "\">" << prefix << "/" << key << "</DataItem>" << std::endl;

    if (lo != nullptr && hi != nullptr) {
        os << std::setw(indent) << " "
           << "</DataItem>" << std::endl;
    }
    return os;
}
void DataEntryXDMF::WriteDataItem(std::string const &url, std::string const &key, const index_box_type &idx_box,
                                  std::shared_ptr<const data::DataEntry> const &data, int indent) {
    auto g_id = H5GroupTryOpen(m_h5_root_, url);
    if (data == nullptr) {
    } else if (auto array = std::dynamic_pointer_cast<const ArrayBase>(data->GetEntity())) {
        XDMFWriteArray(*m_os_, g_id, m_h5_prefix_ + ":" + url, key, &std::get<0>(idx_box)[0], &std::get<1>(idx_box)[0],
                       array, indent + 1);
    } else if (data->type() == data::DataEntry::DN_ARRAY) {
        auto dof = static_cast<int>(data->size());

        *m_os_ << std::setw(indent + 1) << " "
               << R"(<DataItem  ItemType="Function" )";
        *m_os_ << R"(Function="$0)";
        for (int i = 1; i < dof; ++i) { *m_os_ << " ,$" << i; }
        *m_os_ << R"(" Dimensions=")";
        for (int i = 0; i < 3; ++i) { *m_os_ << " " << std::get<1>(idx_box)[i] - std::get<0>(idx_box)[i]; }
        *m_os_ << " " << dof;
        *m_os_ << R"(" >)" << std::endl;

        auto subg_id = H5Gcreate(g_id, key.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (int i = 0; i < dof; ++i) {
            if (auto array = std::dynamic_pointer_cast<const ArrayBase>(data->GetEntity(i))) {
                std::string prefix = m_h5_prefix_;
                prefix += ":";
                prefix += url;
                prefix += "/";
                prefix += key;
                XDMFWriteArray(*m_os_, subg_id, prefix, std::to_string(i), &std::get<0>(idx_box)[0],
                               &std::get<1>(idx_box)[0], array, indent + 2);
            }
        }
        *m_os_ << std::setw(indent + 1) << " "
               << R"(</DataItem> )" << std::endl;
        H5Gclose(subg_id);
    } else {
        UNIMPLEMENTED;
    }
    H5Gclose(g_id);
}
void DataEntryXDMF::WriteAttribute(std::string const &url, std::string const &key, const index_box_type &idx_box,
                                   std::shared_ptr<const data::DataEntry> const &attr, int indent) {
    index_tuple lo{0, 0, 0}, hi{1, 1, 1};
    std::tie(lo, hi) = idx_box;

    auto iform = attr->GetValue<int>("IFORM", NODE);
    if (iform == NODE) { hi += 1; }
    int dof = 1;
    std::string attr_type = "Scalar";
    size_type rank = 0;
    size_type extents[SP_ARRAY_MAX_NDIMS] = {1, 1, 1, 1};
    if (auto p = attr->Get("DOF")) {
        if (auto dof_t = std::dynamic_pointer_cast<const DataLightT<int>>(p->GetEntity())) {
            dof = (dof_t->value());
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
            extents[0] = static_cast<size_type>(dof);
        } else if (auto dof_p = std::dynamic_pointer_cast<const DataLightT<int *>>(p->GetEntity())) {
            attr_type = "Matrix";
            rank = dof_p->extents(extents);
        }
    }
    if (dof == 1 && (iform == EDGE || iform == FACE)) { attr_type = "Vector"; }

    *m_os_ << std::setw(indent) << " "
           << "<Attribute "
           << "Center=\"" << (iform == NODE ? "Node" : "Cell") << "\" "  //
           << "Name=\"" << key << "\" "                                  //
           << "AttributeType=\"" << attr_type << "\" "                   //
           << "IFORM=\"" << iform << "\" "                               //
           << ">" << std::endl;

    WriteDataItem(url, key, std::make_tuple(lo, hi), attr->Get("_DATA_"), indent + 1);

    *m_os_ << std::setw(indent) << " "
           << "</Attribute>" << std::endl;
}
void DataEntryXDMF::WriteParticle(std::string const &url, std::string const &key, const index_box_type &idx_box,
                                  std::shared_ptr<const data::DataEntry> const &data, int indent) {
    *m_os_ << std::setw(indent) << " "
           << "<Grid Name=\"" << key << "  GridType=\"Uniform\"> " << std::endl
           << std::setw(indent + 1) << " "
           << "<Time Value=\"" << 0.00 << "\"  />" << std::endl
           << std::setw(indent + 1) << " "
           << R"(<Topology TopologyType="Polyvertex" NodesPerElement=")" << 30 << "\" />" << std::endl;

    *m_os_ << R"(<Geometry GeometryType="XYZ">
					<DataItem DataType="Float" Dimensions="30 3" Format="HDF">
						TestData.h5:/iter00000000/cells/position
					</DataItem>
				</Geometry>
				<Attribute AttributeType="Scalar" Center="Node" Name="leuk_type">
					<DataItem DataType="Int" Dimensions="30 1" Format="HDF">
						TestData.h5:/iter00000000/cells/type
					</DataItem>
				</Attribute>
				<Attribute AttributeType="Vector" Center="Node"
Name="leuk_polarization">
					<DataItem DataType="Float" Dimensions="30 3" Format="HDF">
						TestData.h5:/iter00000000/cells/polarization
					</DataItem>
				</Attribute>

)";
    *m_os_ << std::setw(indent) << " "
           << "</Grid>" << std::endl;
}
void XDMFGeometryCurvilinear(DataEntryXDMF *self, std::string const &prefix, const index_box_type &idx_box,
                             std::shared_ptr<const DataEntry> const &chart,
                             std::shared_ptr<const data::DataEntry> const &coord, int indent) {
    index_tuple lo, hi;
    std::tie(lo, hi) = idx_box;
    hi += 1;
    *self->m_os_ << std::setw(indent) << " "
                 << R"(<Topology TopologyType="3DSMesh" Dimensions=")" <<  //
        hi[0] - lo[0] << " " <<                                            //
        hi[1] - lo[1] << " " <<                                            //
        hi[2] - lo[2] << "\" />" << std::endl;
    *self->m_os_ << std::setw(indent) << " "
                 << R"(<Geometry GeometryType="XYZ">)" << std::endl;

    self->WriteDataItem(prefix, "_XYZ_", std::make_tuple(lo, hi), coord->Get("_DATA_"), indent + 1);
    *self->m_os_ << std::setw(indent) << " "
                 << "</Geometry>" << std::endl;
}

/**
 *  @ref [Paraview] Fwd: Odd behavior of XDMF files with 3DCORECTMesh
 *
 *  @quota
 *

 */
void XDMFGeometryRegular(DataEntryXDMF *self, const index_box_type &idx_box,
                         std::shared_ptr<const DataEntry> const &chart, int indent = 0) {
    index_tuple lo, hi;
    std::tie(lo, hi) = idx_box;
    hi += 1;

    // @NOTE ParaView and VisIt parser 3DCORectMesh in different way, but 3DRectMesh is OK.
    *self->m_os_ << std::setw(indent) << " "
                 << R"(<Topology TopologyType="3DRectMesh" Dimensions=")" << hi[0] - lo[0] << " " << hi[1] - lo[1]
                 << " " << hi[2] - lo[2] << "\" />" << std::endl;

    auto origin = chart->GetValue<point_type>("Origin", point_type{0, 0, 0});
    auto dxdydz = chart->GetValue<point_type>("Scale", point_type{1, 1, 1});
    *self->m_os_ << std::setw(indent) << " "
                 << R"(<Geometry GeometryType="ORIGIN_DXDYDZ"> )" << std::endl
                 << std::setw(indent + 1) << " "
                 << R"(<DataItem Format="XML" Dimensions="3"> )"  // NOTE: inverse xyz order. bug in  libXDMF?
                 << origin[2] + lo[2] * dxdydz[2] << " "          //
                 << origin[1] + lo[1] * dxdydz[1] << " "          //
                 << origin[0] + lo[0] * dxdydz[0]                 //
                 << " </DataItem>" << std::endl
                 << std::setw(indent + 1) << " "
                 << R"(<DataItem Format="XML" Dimensions="3"> )" << dxdydz[2] << " " << dxdydz[1] << " " << dxdydz[0]
                 << " </DataItem>" << std::endl;

    *self->m_os_ << std::setw(indent) << " "
                 << "</Geometry>" << std::endl;
}
int DataEntryXDMF::Flush() {
    int success = SP_FAILED;

    //    auto attrs = this->Get("Attributes");
    int indent = 2;
    if (auto atlas = this->Get("Atlas")) {
        auto chart = atlas->Get("Chart");
        if (auto patches = atlas->Get("Patches")) {
            patches->Foreach([&](std::string const &k, std::shared_ptr<const data::DataEntry> const &patch) {
                auto attrs = patch->Get("Attributes");
                auto mblk = patch->Get("MeshBlock");
                auto guid = mblk->GetValue<id_type>("GUID");

                index_box_type idx_box{mblk->GetValue<index_tuple>("LowIndex"),
                                       mblk->GetValue<index_tuple>("HighIndex")};
                std::get<0>(idx_box) -= 1;  // add ghost cell
                std::get<1>(idx_box) += 1;

                *m_os_ << std::setw(indent) << " "
                       << "<Grid Name=\"" << guid << "\" Level=\"" << patch->GetValue<int>("Level", 0) << "\">"
                       << std::endl;

                if (auto coord = attrs->Get("_COORDINATES_")) {
                    XDMFGeometryCurvilinear(this, "/Patches/" + std::to_string(guid), idx_box, chart, coord,
                                            indent + 1);
                } else {
                    XDMFGeometryRegular(this, idx_box, chart, indent + 1);
                }

                attrs->Foreach([&](std::string const &s, std::shared_ptr<const data::DataEntry> const &d) {
                    if (d->GetValue<int>("IFORM") != FIBER) {
                        WriteAttribute("/Patches/" + std::to_string(guid), s, idx_box, d, indent + 1);
                    }
                });
                *m_os_ << std::setw(indent) << " "
                       << "</Grid>" << std::endl;

                attrs->Foreach([&](std::string const &s, std::shared_ptr<const data::DataEntry> const &d) {
                    if (d->GetValue<int>("IFORM") == FIBER) {
                        WriteParticle("/Patches/" + std::to_string(guid), s, idx_box, d, indent + 1);
                    }
                });
                return 1;
            });
        }
    }

    // Gather String
    std::string grid_str = m_os_->str();

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
        out_file.open(m_prefix_ + ".xmf", std::ios_base::trunc);
        VERBOSE << std::setw(20) << "Write XDMF : " << m_prefix_ << ".xmf";

        out_file << R"(<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.3">
<Domain>
<Grid CollectionType="Spatial" GridType="Collection" Name="Collection">)";
        if (auto time = this->Get("Time")) {
            if (auto t = std::dynamic_pointer_cast<const DataLightT<Real>>(time->GetEntity())) {
                out_file << std::endl << "  <Time Value=\"" << t->value() << "\"/>" << std::endl;
            }
        }
        out_file << std::setw(indent) << " " << grid_str;
        out_file << R"(</Grid>
</Domain>
</Xdmf>)";

        out_file.close();
    }
    return success;
}

}  // namespace data{
}  // namespace simpla