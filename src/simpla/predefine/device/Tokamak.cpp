//
// Created by salmon on 17-7-9.
//
#include "Tokamak.h"

#include <simpla/algebra/nTuple.h>
#include <simpla/data/Data.h>
#include <simpla/geometry/Polygon.h>
#include <simpla/geometry/Polyline2d.h>
#include <simpla/geometry/Revolution.h>
#include <simpla/numeric/Interpolation.h>
#include <simpla/numeric/find_root.h>
#include <simpla/utilities/Constants.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Log.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace simpla {

struct Tokamak::pimpl_s {
    typedef Interpolation<LinearInterpolation, Real, Real> inter_type;
    typedef MultiDimensionInterpolation<BiLinearInterpolation, Real> inter2d_type;

    //	coordinates_type rzmin_;
    //	coordinates_type rzmax_;

    //	inter_type fpol_; //!< Poloidal current function in m-T $F=RB_T$ on flux grid
    //	inter_type pres_;//!< Plasma pressure in $nt/m^2$ on uniform flux grid
    //	inter_type ffprim_;//!< \f$FF^\prime(\psi)\f$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
    //	inter_type pprim_;//!< $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid

    inter2d_type m_psirz_;  //!< Poloidal flux in Webber/rad on the rectangular grid points

    //	inter_type qpsi_;//!< q values on uniform flux grid from axis to boundary
    std::shared_ptr<geometry::Polyline2d> m_rzbbb_;  //!< R,Z of boundary points in meter
    std::shared_ptr<geometry::Polyline2d> m_rzlim_;  //!< R,Z of surrounding limiter contour in meter
    std::map<std::string, inter_type> m_profile_;
    //    bool flux_surface(Real psi_j, size_t M, point_type *res, Real resoluton = 0.001);

    Real m_phi0_ = 0.0, m_phi1_ = TWOPI;
};

Tokamak::Tokamak(std::string const &url) : m_pimpl_(new pimpl_s) { ReadGFile(url); }
Tokamak::~Tokamak() { delete m_pimpl_; }
void Tokamak::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Tokamak::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}

void Tokamak::ReadGFile(std::string const &fname) {
    std::ifstream inFileStream_(fname);
    geometry::Axis axis;
    m_pimpl_->m_rzbbb_ = geometry::Polyline2d::New(axis);
    m_pimpl_->m_rzlim_ = geometry::Polyline2d::New(axis);
    if (!inFileStream_.is_open()) {
        THROW_EXCEPTION_RUNTIME_ERROR("File " + fname + " is not opend!");
        return;
    }

    LOGGER << "Load GFile : [" << fname << "]";

    char str_buff[50];

    int nw;        // Number of horizontal R grid points
    int nh;        // Number of vertical Z grid points
    double simag;  // Poloidal flux at magnetic axis in Weber / rad
    double sibry;  // Poloidal flux at the plasma boundary in Weber / rad
    int idum;
    double xdum;

    size_type m_nr_ = 0, m_nz_ = 0;
    Real m_rmin_ = 0.0, m_zmin_ = 0.0;
    Real m_rmax_ = 0.0, m_zmax_ = 0.0;
    bool is_valid_ = false;
    std::string m_desc_;
    //	size_t nw;//!< Number of horizontal R grid  points
    //	size_t nh;//!< Number of vertical Z grid points
    Real m_rdim_ = 0.0;    //!< Horizontal dimension in meter of computational box
    Real m_zdim_ = 0.0;    //!< Vertical dimension in meter of computational box
    Real m_rleft_ = 0.0;   //!< Minimum R in meter of rectangular computational box
    Real m_zmid_ = 0.0;    //!< Z of center of computational box in meter
    Real m_rmaxis_ = 1.0;  //!< R of magnetic axis in meter
    Real m_zmaxis = 1.0;   //!< Z of magnetic axis in meter
    //	Real simag;//!< Poloidal flux at magnetic axis in Weber / rad
    //	Real sibry;//!< Poloidal flux at the plasma boundary in Weber / rad
    Real m_rcenter_ = 0.5;  //!< R in meter of  vacuum toroidal magnetic field BCENTR
    Real m_bcenter_ = 0.5;  //!< Vacuum toroidal magnetic field in Tesla at RCENTR
    Real m_current_ = 1.0;  //!< Plasma current in Ampere

    inFileStream_.get(str_buff, 48);

    db()->SetValue("Description", std::string(str_buff));

    inFileStream_ >> std::setw(4) >> idum >> nw >> nh;

    inFileStream_ >> std::setw(16) >> m_rdim_ >> m_zdim_ >> m_rcenter_ >> m_rleft_ >> m_zmid_ >> m_rmaxis_ >>
        m_zmaxis >> simag >> sibry >> m_bcenter_ >> m_current_ >> simag >> xdum >> m_rmaxis_ >> xdum >> m_zmaxis >>
        xdum >> sibry >> xdum >> xdum;

    m_rmin_ = m_rleft_;
    m_rmax_ = m_rleft_ + m_rdim_;
    m_zmin_ = m_zmid_ - m_zdim_ / 2;
    m_zmax_ = m_zmid_ + m_zdim_ / 2;

    m_nr_ = static_cast<size_type>(nw);
    m_nz_ = static_cast<size_type>(nh);

    size_type dims[2] = {m_nr_, m_nz_};
    Real rzmin[2] = {m_rmin_, m_zmin_};
    Real rzmax[2] = {m_rmax_, m_zmax_};

    typedef MultiDimensionInterpolation<BiLinearInterpolation, Real> inter2d_type;

    inter2d_type(dims, rzmin, rzmax).swap(m_pimpl_->m_psirz_);

#define INPUT_VALUE(_NAME_)                                                                                   \
    for (int s = 0; s < nw; ++s) {                                                                            \
        double y;                                                                                             \
        inFileStream_ >> std::setw(16) >> y;                                                                  \
        m_pimpl_->m_profile_[_NAME_].data().emplace(static_cast<double>(s) / static_cast<double>(nw - 1), y); \
    }

    INPUT_VALUE("fpol");
    INPUT_VALUE("pres");
    INPUT_VALUE("ffprim");
    INPUT_VALUE("pprim");

    for (int j = 0; j < nh; ++j)
        for (int i = 0; i < nw; ++i) {
            double v;
            inFileStream_ >> std::setw(16) >> v;
            m_pimpl_->m_psirz_[i + j * nw] = (v - simag) / (sibry - simag);  // Normalize Poloidal flux
        }

    INPUT_VALUE("qpsi");

#undef INPUT_VALUE

    size_t n_bbbs, n_limitr;
    inFileStream_ >> std::setw(5) >> n_bbbs >> n_limitr;

    m_pimpl_->m_rzbbb_->data().resize(n_bbbs);
    m_pimpl_->m_rzlim_->data().resize(n_limitr);

    inFileStream_ >> std::setw(16) >> m_pimpl_->m_rzbbb_->data();
    inFileStream_ >> std::setw(16) >> m_pimpl_->m_rzlim_->data();

    ReadProfile(fname + "_profiles.txt");
}
void Tokamak::WriteGFile(std::string const &) const { UNIMPLEMENTED; }

void Tokamak::ReadProfile(std::string const &fname) {
    LOGGER << "Load GFile Profiles: [" << fname << "]";

    std::ifstream inFileStream_(fname);
    if (!inFileStream_.is_open()) { THROW_EXCEPTION_RUNTIME_ERROR("File " + fname + " is not opened!"); }
    std::string line;
    std::getline(inFileStream_, line);
    std::vector<std::string> names;
    {
        std::stringstream lineStream(line);
        while (lineStream) {
            std::string t;
            lineStream >> t;
            if (t.empty()) names.push_back(t);
        };
    }

    while (inFileStream_) {
        auto it = names.begin();
        auto ie = names.end();
        double psi;
        inFileStream_ >> psi;  /// \note assume first row is psi
        *it = psi;

        for (++it; it != ie; ++it) {
            double value;
            inFileStream_ >> value;
            m_pimpl_->m_profile_[*it].data().emplace(psi, value);
        }
    }
    std::string profile_list = "psi,B";

    for (auto const &item : m_pimpl_->m_profile_) { profile_list += " , " + item.first; }

    LOGGER << "GFile is ready! Profile={" << profile_list << "}";
}

Real Tokamak::psi(Real R, Real Z) const { return m_pimpl_->m_psirz_(R, Z); }
nTuple<Real, 2> Tokamak::grad_psi(Real R, Real Z) const { return m_pimpl_->m_psirz_.grad(R, Z); }
Real Tokamak::profile(std::string const &name, Real p_psi) const { return m_pimpl_->m_profile_[name](p_psi); }
std::function<Real(point_type const &x)> Tokamak::GetAttribute(std::string const &k_name) const {
    std::function<Real(point_type const &x)> res = nullptr;
    auto it = m_pimpl_->m_profile_.find(k_name);
    if (it != m_pimpl_->m_profile_.end()) {
        auto &fun = it->second;
        res = [&](point_type const &x) { return fun(psi(x)); };
    }
    return res;
}
std::function<Real(point_type const &)> Tokamak::profile(std::string const &attr_name) const {
    std::function<Real(point_type const &)> res = nullptr;

    if (attr_name == "psi") {
        res = [&](point_type const &x) { return psi(x); };
    } else if (attr_name == "JT") {
        res = [&](point_type const &x) { return JT(x[0], x[1]); };
    } else {
        res = GetAttribute(attr_name);
    }

    return res;
};
std::function<Vec3(point_type const &)> Tokamak::B0() const {
    return [&](point_type const &x) { return B(x); };
};
std::shared_ptr<geometry::GeoObject> Tokamak::Limiter() const {
    //    BRepBuilderAPI_MakeWire wireMaker;
    //    auto num = boundary()->data().size();
    //    Handle(TColgp_HArray1OfPnt) gp_array = new TColgp_HArray1OfPnt(1, static_cast<Standard_Integer>(num));
    //    auto const &points = boundary()->data();
    //    for (size_type s = 0; s < num - 1; ++s) { gp_array->SetValue(s + 1, gp_Pnt(points[s][0], 0, points[s][1])); }
    //    GeomAPI_Interpolate sp(gp_array, true, Precision::Confusion());
    //    sp.Perform();
    //    wireMaker.Add(BRepBuilderAPI_MakeEdge(sp.Curve()));
    //    gp_Ax1 axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
    //    BRepBuilderAPI_MakeFace myBoundaryFaceProfile(wireMaker.Wire(), true);
    //    BRepPrimAPI_MakeRevol revol(myBoundaryFaceProfile.Face(), axis);
    return geometry::Revolution::New(geometry::Polygon::New(), point_type{0, 0, 0}, vector_type{0, 0, 1},
                                     m_pimpl_->m_phi0_, m_pimpl_->m_phi1_);
}
std::shared_ptr<geometry::GeoObject> Tokamak::Boundary() const {
    //    BRepBuilderAPI_MakePolygon polygonMaker;
    //    for (auto const &p : limiter()->data()) { polygonMaker.Add(gp_Pnt(p[0], 0, p[1])); }
    //    gp_Ax1 axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
    //    BRepBuilderAPI_MakeFace myLimterFaceProfile(polygonMaker.Wire());
    //    BRepPrimAPI_MakeRevol myLimiter(myLimterFaceProfile.Face(), axis);
    //    return geometry::GeoObjectOCE::New(myLimiter.Shape());
    return geometry::Revolution::New(geometry::Polygon::New(), point_type{0, 0, 0}, vector_type{0, 0, 1},
                                     m_pimpl_->m_phi0_, m_pimpl_->m_phi1_);
}

}  // namespace simpla {
