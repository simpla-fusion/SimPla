/**
 *  @file  Geqdsk.cpp
 *
 *  created on: 2013-11-29
 *      Author: salmon
 */

#include "GEqdsk.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <utility>

#include "../physics/Constants.h"
#include "../gtl/utilities/log.h"
#include "../gtl/utilities/pretty_stream.h"
#include "../gtl/ntuple.h"
#include "../gtl/ntuple_ext.h"


#include "../geometry/polygon.h"
#include "../numeric/find_root.h"
#include "../numeric/interpolation.h"

#ifndef NO_XDMF

#include "../io/XDMFIO.h"

#endif
namespace simpla
{
constexpr int GEqdsk::PhiAxis;
constexpr int GEqdsk::RAxis;
constexpr int GEqdsk::ZAxis;

struct GEqdsk::pimpl_s
{
    typedef Interpolation<LinearInterpolation, Real, Real> inter_type;

    typedef MultiDimesionInterpolation<BiLinearInterpolation, Real> inter2d_type;

    nTuple<int, 3> m_dims_{1, 1, 1};
    point_type m_rzmin_;
    point_type m_rzmax_;

    bool is_valid_ = false;
    std::string m_desc_;
//	size_t nw;//!< Number of horizontal R grid  points
//	size_t nh;//!< Number of vertical Z grid points
    Real m_rdim_; //!< Horizontal dimension in meter of computational box
    Real m_zdim_; //!< Vertical dimension in meter of computational box
    Real m_rleft_; //!< Minimum R in meter of rectangular computational box
    Real m_zmid_; //!< Z of center of computational box in meter
    Real m_rmaxis_ = 1.0; //!< R of magnetic axis in meter
    Real m_zmaxis = 1.0; //!< Z of magnetic axis in meter
//	Real simag;//!< Poloidal flux at magnetic axis in Weber / rad
//	Real sibry;//!< Poloidal flux at the plasma boundary in Weber / rad
    Real m_rcenter_ = 0.5; //!< R in meter of  vacuum toroidal magnetic field BCENTR
    Real m_bcenter_ = 0.5; //!< Vacuum toroidal magnetic field in Tesla at RCENTR
    Real m_current_ = 1.0; //!< Plasma current in Ampere

//	coordinates_type rzmin_;
//	coordinates_type rzmax_;

//	inter_type fpol_; //!< Poloidal current function in m-T $F=RB_T$ on flux grid
//	inter_type pres_;//!< Plasma pressure in $nt/m^2$ on uniform flux grid
//	inter_type ffprim_;//!< \f$FF^\prime(\psi)\f$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
//	inter_type pprim_;//!< $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid

    inter2d_type m_psirz_; //!< Poloidal flux in Webber/rad on the rectangular grid points

//	inter_type qpsi_;//!< q values on uniform flux grid from axis to boundary



    geometry::Polygon<2> m_rzbbb_; //!< R,Z of boundary points in meter
    geometry::Polygon<2> m_rzlim_; //!< R,Z of surrounding limiter contour in meter

    std::map<std::string, inter_type> m_profile_;

    void load(std::string const &fname);

    void load_profile(std::string const &fname);

//    bool flux_surface(Real psi_j, size_t M, point_type *res, Real resoluton = 0.001);

    void write(std::string const &);
};

void GEqdsk::load(std::string const &fname)
{
    m_pimpl_->load(fname);

}

nTuple<int, 3> const &GEqdsk::dimensions() const
{
    return m_pimpl_->m_dims_;
};

void GEqdsk::pimpl_s::load(std::string const &fname)
{


    std::ifstream inFileStream_(fname);

    if (!inFileStream_.is_open())
    {
        THROW_EXCEPTION_RUNTIME_ERROR("File " + fname + " is not opend!");
        return;
    }

    LOGGER << "Load GFile : [" << fname << "]" << std::endl;

    int nw; //Number of horizontal R grid points
    int nh; //Number of vertical Z grid points
    double simag; // Poloidal flux at magnetic axis in Weber / rad
    double sibry; // Poloidal flux at the plasma boundary in Weber / rad
    int idum;
    double xdum;

    char str_buff[50];

    inFileStream_.get(str_buff, 48);

    m_desc_ = std::string(str_buff);

    inFileStream_ >> std::setw(4) >> idum >> nw >> nh;

    inFileStream_ >> std::setw(16)

    >> m_rdim_ >> m_zdim_ >> m_rcenter_ >> m_rleft_ >> m_zmid_

    >> m_rmaxis_ >> m_zmaxis >> simag >> sibry >> m_bcenter_

    >> m_current_ >> simag >> xdum >> m_rmaxis_ >> xdum

    >> m_zmaxis >> xdum >> sibry >> xdum >> xdum;

    m_rzmin_[RAxis] = m_rleft_;
    m_rzmax_[RAxis] = m_rleft_ + m_rdim_;
    m_rzmin_[ZAxis] = m_zmid_ - m_zdim_ / 2;
    m_rzmax_[ZAxis] = m_zmid_ + m_zdim_ / 2;
    m_rzmin_[PhiAxis] = 0;
    m_rzmax_[PhiAxis] = TWOPI;

    m_dims_[RAxis] = nw;
    m_dims_[ZAxis] = nh;
    m_dims_[PhiAxis] = 1;

    inter2d_type(m_dims_, m_rzmin_, m_rzmax_, PhiAxis).swap(m_psirz_);

#define INPUT_VALUE(_NAME_)                                               \
    for (int s = 0; s < nw; ++s)                                          \
    {                                                                     \
        double y;                                                         \
        inFileStream_ >> std::setw(16) >> y;                              \
        m_profile_[ _NAME_ ].data().emplace(                              \
          static_cast<double>(s)                                          \
              /static_cast<double>(nw-1), y );                            \
    }                                                                     \

    INPUT_VALUE("fpol");
    INPUT_VALUE("pres");
    INPUT_VALUE("ffprim");
    INPUT_VALUE("pprim");

    for (int j = 0; j < nh; ++j)
        for (int i = 0; i < nw; ++i)
        {
            double v;
            inFileStream_ >> std::setw(16) >> v;
            m_psirz_[i + j * nw] = (v - simag) / (sibry - simag); // Normalize Poloidal flux
        }

    INPUT_VALUE("qpsi");

#undef INPUT_VALUE

    size_t n_bbbs, n_limitr;
    inFileStream_ >> std::setw(5) >> n_bbbs >> n_limitr;


    m_rzbbb_.data().resize(n_bbbs);
    m_rzlim_.data().resize(n_limitr);

    inFileStream_ >> std::setw(16) >> m_rzbbb_.data();
    inFileStream_ >> std::setw(16) >> m_rzlim_.data();

    m_rzbbb_.deploy();
    m_rzlim_.deploy();


    load_profile(fname + "_profiles.txt");

}

void GEqdsk::load_profile(std::string const &fname)
{
    m_pimpl_->load_profile(fname);
}

void GEqdsk::pimpl_s::load_profile(std::string const &fname)
{
    LOGGER << "Load GFile Profiles: [" << fname << "]" << std::endl;

    std::ifstream inFileStream_(fname);

    if (!inFileStream_.is_open())
    {
        THROW_EXCEPTION_RUNTIME_ERROR("File " + fname + " is not opend!");
    }

    std::string line;

    std::getline(inFileStream_, line);

    std::vector<std::string> names;
    {
        std::stringstream lineStream(line);

        while (lineStream)
        {
            std::string t;
            lineStream >> t;
            if (t != "")
                names.push_back(t);
        };
    }

    while (inFileStream_)
    {
        auto it = names.begin();
        auto ie = names.end();
        double psi;
        inFileStream_ >> psi;        /// \note assume first row is psi
        *it = psi;

        for (++it; it != ie; ++it)
        {
            double value;
            inFileStream_ >> value;
            m_profile_[*it].data().emplace(psi, value);

        }
    }
    std::string profile_list = "psi,B";

    for (auto const &item : m_profile_)
    {
        profile_list += " , " + item.first;
    }

    LOGGER << "GFile is ready! Profile={" << profile_list << "}" << std::endl;

    is_valid_ = true;

}

Real GEqdsk::B0() const { return m_pimpl_->m_bcenter_; }

std::ostream &GEqdsk::print(std::ostream &os)
{
    std::cout << "--" << m_pimpl_->m_desc_ << std::endl;

//	std::cout << "nw" << "\t= " << nw
//			<< "\t--  Number of horizontal R grid  points" << std::endl;
//
//	std::cout << "nh" << "\t= " << nh << "\t-- Number of vertical Z grid points"
//			<< std::endl;
//
//	std::cout << "rdim" << "\t= " << rdim
//			<< "\t-- Horizontal dimension in meter of computational box                   "
//			<< std::endl;
//
//	std::cout << "zdim" << "\t= " << zdim
//			<< "\t-- Vertical dimension in meter of computational box                   "
//			<< std::endl;

    std::cout << "rcentr" << "\t= " << m_pimpl_->m_rcenter_
    << "\t--                                                                    " << std::endl;

//	std::cout << "rleft" << "\t= " << rleft
//			<< "\t-- Minimum R in meter of rectangular computational box                "
//			<< std::endl;
//
//	std::cout << "zmid" << "\t= " << zmid
//			<< "\t-- Z of center of computational box in meter                          "
//			<< std::endl;

    std::cout << "rmaxis" << "\t= " << m_pimpl_->m_rmaxis_
    << "\t-- R of magnetic axis in meter                                        " << std::endl;

    std::cout << "rmaxis" << "\t= " << m_pimpl_->m_zmaxis
    << "\t-- Z of magnetic axis in meter                                        " << std::endl;

//	std::cout << "simag" << "\t= " << simag
//			<< "\t-- poloidal flus ax magnetic axis in Weber / rad                      "
//			<< std::endl;
//
//	std::cout << "sibry" << "\t= " << sibry
//			<< "\t-- Poloidal flux at the plasma boundary in Weber / rad                "
//			<< std::endl;

    std::cout << "rcentr" << "\t= " << m_pimpl_->m_rcenter_
    << "\t-- R in meter of  vacuum toroidal magnetic field BCENTR               " << std::endl;

    std::cout << "bcentr" << "\t= " << m_pimpl_->m_bcenter_
    << "\t-- Vacuum toroidal magnetic field in Tesla at RCENTR                  " << std::endl;

    std::cout << "current" << "\t= " << m_pimpl_->m_current_
    << "\t-- Plasma current in Ampere                                          " << std::endl;

//	std::cout << "fpol" << "\t= "
//			<< "\t-- Poloidal current function in m-T<< $F=RB_T$ on flux grid           "
//			<< std::endl << fpol_.data() << std::endl;
//
//	std::cout << "pres" << "\t= "
//			<< "\t-- Plasma pressure in $nt/m^2$ on uniform flux grid                   "
//			<< std::endl << pres_.data() << std::endl;
//
//	std::cout << "ffprim" << "\t= "
//			<< "\t-- $FF^\\prime(\\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid     "
//			<< std::endl << ffprim_.data() << std::endl;
//
//	std::cout << "pprim" << "\t= "
//			<< "\t-- $P^\\prime(\\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid    "
//			<< std::endl << pprim_.data() << std::endl;
//
//	std::cout << "psizr"
//			<< "\t-- Poloidal flus in Webber/rad on the rectangular grid points         "
//			<< std::endl << m_psirz_.data() << std::endl;
//
//	std::cout << "qpsi" << "\t= "
//			<< "\t-- q values on uniform flux grid from axis to boundary                "
//			<< std::endl << qpsi_.data() << std::endl;
//
//	std::cout << "nbbbs" << "\t= " << nbbbs
//			<< "\t-- Number of boundary points                                          "
//			<< std::endl;
//
//	std::cout << "limitr" << "\t= " << limitr
//			<< "\t-- Number of limiter points                                           "
//			<< std::endl;
//
//	std::cout << "rzbbbs" << "\t= "
//			<< "\t-- R of boundary points in meter                                      "
//			<< std::endl << rzbbb_ << std::endl;
//
//	std::cout << "rzlim" << "\t= "
//			<< "\t-- R of surrounding limiter contour in meter                          "
//			<< std::endl << rzlim_ << std::endl;

    return os;
}
//
//bool GEqdsk::flux_surface(double psi_j, size_t M, point_type *res,
//                          double resolution)
//{
//    return m_pimpl_->flux_surface(psi_j, M, res, resolution);
//}
//
//bool GEqdsk::pimpl_s::flux_surface(double psi_j, size_t M, point_type *res,
//                                   double resolution)
//{
//
//    //FIXME need check
//    Real success = 0;
//
//    nTuple<double, 3> center;
//
//    center[PhiAxis] = 0;
//    center[RAxis] = m_rcenter_;
//    center[ZAxis] = m_zmid_;
//
//    nTuple<double, 3> drz;
//
//    drz[PhiAxis] = 0;
//
//    std::function<double(nTuple<double, 3> const &)> fun =
//            [this](nTuple<double, 3> const &x) -> double
//            {
//                return m_psirz_(x[RAxis], x[ZAxis]);
//            };
//
//    for (int i = 0; i < M; ++i)
//    {
//        double theta = static_cast<double>(i) * TWOPI / static_cast<double>(M);
//
//        drz[RAxis] = std::cos(theta);
//        drz[ZAxis] = std::sin(theta);
//
//        nTuple<double, 3> rmax;
//        nTuple<double, 3> t;
//
//        rmax = center
//               + drz * std::sqrt(m_rdim_ * m_rdim_ + m_zdim_ * m_zdim_) * 0.5;
//
//        success = m_rzbbb_->nearest_point(center, &rmax);
//
//        if (success > 0)
//        {
//            THROW_EXCEPTION_RUNTIME_ERROR(
//                    "Illegal Geqdsk configuration: RZ-center is out of the boundary (rzbbb)!  ");
//        }
//
//        nTuple<double, 3> t2 = center + (rmax - center) * 0.1;
//
//        res[i] = rmax;
//        success = find_root(fun, psi_j, t2, &res[i], resolution);
//
//        if (!success)
//        {
//            WARNING << "Construct flux surface failed!" << "at theta = "
//            << theta << " psi = " << psi_j;
//            break;
//        }
//
//    }
//
//    return success == 0;
//
//}


GEqdsk::GEqdsk() : m_pimpl_(new pimpl_s) { }

GEqdsk::~GEqdsk() { }

std::string const &GEqdsk::description() const { return m_pimpl_->m_desc_; }


geometry::GeoObject const &GEqdsk::boundary() const
{
    return dynamic_cast<geometry::GeoObject const &>(m_pimpl_->m_rzbbb_);
}

geometry::GeoObject const &GEqdsk::limiter() const
{
    return dynamic_cast<geometry::GeoObject const &>(m_pimpl_->m_rzlim_);
}

Real GEqdsk::psi(Real R, Real Z) const
{
    return m_pimpl_->m_psirz_(R, Z);
}

Vec3 GEqdsk::grad_psi(Real R, Real Z) const
{
    return m_pimpl_->m_psirz_.grad(R, Z);
}

Real GEqdsk::profile(std::string const &name, Real p_psi) const
{
    return m_pimpl_->m_profile_[name](p_psi);
}

GEqdsk::point_type GEqdsk::magnetic_axis() const
{
    return point_type{m_pimpl_->m_rmaxis_, m_pimpl_->m_zmaxis, 0};
}

std::tuple<typename GEqdsk::point_type, typename GEqdsk::point_type> GEqdsk::box() const
{
    return std::make_tuple(m_pimpl_->m_rzmin_, m_pimpl_->m_rzmax_);
}

void GEqdsk::write(std::string const &url)
{
    m_pimpl_->write(url);
}

void GEqdsk::pimpl_s::write(std::string const &url)
{
#ifndef NO_XDMF

    typedef nTuple<Real, 3> point_type;
    XdmfDOM dom;
    XdmfRoot root;
    root.SetDOM(&dom);
    root.SetVersion(2.0);
    root.Build();

    XdmfDomain domain;


    root.Insert(&domain);

    {
        XdmfGrid grid;
        domain.Insert(&grid);

        grid.SetName("G-Eqdsk");
        grid.SetGridType(XDMF_GRID_UNIFORM);
        grid.GetTopology()->SetTopologyTypeFromString("2DCoRectMesh");

        XdmfInt64 dims[2] = {static_cast<XdmfInt64>(m_dims_[1]), static_cast<XdmfInt64>(m_dims_[0])};
        grid.GetTopology()->GetShapeDesc()->SetShape(2, dims);

        grid.GetGeometry()->SetGeometryTypeFromString("Origin_DxDy");
        grid.GetGeometry()->SetOrigin(m_rzmin_[1], m_rzmin_[0], 0);
        grid.GetGeometry()->SetDxDyDz((m_rzmax_[1] - m_rzmin_[1]) / static_cast<Real>(m_dims_[1] - 1),
                                      (m_rzmax_[0] - m_rzmin_[0]) / static_cast<Real>(m_dims_[0] - 1), 0);

        XdmfAttribute myAttribute;
        grid.Insert(&myAttribute);

        myAttribute.SetName("Psi");
        myAttribute.SetAttributeTypeFromString("Scalar");
        myAttribute.SetAttributeCenterFromString("Node");

        XdmfDataItem data;
        myAttribute.Insert(&data);

        io::InsertDataItem(&data, 2, dims, &(m_psirz_.data().get()[0]), url + ".h5:/Psi");
        grid.Build();
    }
    {
        XdmfGrid grid;
        domain.Insert(&grid);
        grid.SetName("Boundary");
        grid.SetGridType(XDMF_GRID_UNIFORM);
        grid.GetTopology()->SetTopologyTypeFromString("POLYLINE");

        XdmfInt64 dims[2] = {static_cast<XdmfInt64>(m_rzbbb_.data().size()), 2};
        grid.GetTopology()->GetShapeDesc()->SetShape(2, dims);
        grid.GetTopology()->Set("NodesPerElement", "2");
        grid.GetTopology()->SetNumberOfElements(static_cast<XdmfInt64>(m_rzbbb_.data().size()));

        XdmfDataItem *data = new XdmfDataItem;

        grid.GetTopology()->Insert(data);

        io::InsertDataItemWithFun(data, 2, dims, [&](XdmfInt64 *d) -> unsigned int
                              {
                                  return static_cast< unsigned int>( d[1] == 0 ? d[0] : (d[0] + 1) % dims[0]);
                              },

                              url + ".h5:/Boundary/Topology");

        grid.GetGeometry()->SetGeometryTypeFromString("XYZ");

        data = new XdmfDataItem;
        data->SetHeavyDataSetName((url + ".h5:/Boundary/Points").c_str());

        grid.GetGeometry()->Insert(data);

        XdmfArray *points = grid.GetGeometry()->GetPoints();

        dims[1] = 3;
        points->SetShape(2, dims);

        XdmfInt64 s = 0;
        for (auto const &v : m_rzbbb_.data())
        {
            points->SetValue(s * 3, 0);
            points->SetValue(s * 3 + 1, v[0]);
            points->SetValue(s * 3 + 2, v[1]);

            ++s;
        }

        grid.Build();
    }
    {
        XdmfGrid grid;
        domain.Insert(&grid);
        grid.SetName("Limter");
        grid.SetGridType(XDMF_GRID_UNIFORM);
        grid.GetTopology()->SetTopologyTypeFromString("POLYLINE");

        XdmfInt64 dims[2] = {static_cast<XdmfInt64>(m_rzbbb_.data().size()), 2};
        grid.GetTopology()->GetShapeDesc()->SetShape(2, dims);
        grid.GetTopology()->Set("NodesPerElement", "2");
        grid.GetTopology()->SetNumberOfElements(static_cast<XdmfInt64>(m_rzbbb_.data().size()));

        XdmfDataItem *data = new XdmfDataItem;

        grid.GetTopology()->Insert(data);

        io::InsertDataItemWithFun(data, 2, dims, [&](XdmfInt64 *d) -> unsigned int
                              {
                                  return static_cast<unsigned int>(d[1] == 0 ? d[0] : (d[0] + 1) % dims[0]);
                              },

                              url + ".h5:/Limter/Topology");

        grid.GetGeometry()->SetGeometryTypeFromString("XYZ");

        data = new XdmfDataItem;
        data->SetHeavyDataSetName((url + ".h5:/Limter/Points").c_str());

        grid.GetGeometry()->Insert(data);

        XdmfArray *points = grid.GetGeometry()->GetPoints();

        dims[1] = 3;
        points->SetShape(2, dims);

        XdmfInt64 s = 0;
        for (auto const &v : m_rzbbb_.data())
        {
            points->SetValue(s * 3, 0);
            points->SetValue(s * 3 + 1, v[0]);
            points->SetValue(s * 3 + 2, v[1]);

            ++s;
        }

        grid.Build();
    }

//		root.Build();
    std::ofstream ss(url + ".xmf");
    ss << dom.Serialize() << std::endl;

#endif // NO_XDMF
}
}  // namespace simpla

