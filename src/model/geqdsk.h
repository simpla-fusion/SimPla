/*
 * read_geqdsk.h
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#ifndef GEQDSK_H_
#define GEQDSK_H_


#include <iostream>
#include <string>
#include <vector>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../numeric/interpolation.h"
namespace simpla
{

/**
 * @ref http://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
 */
class GEqdsk
{

public:

	typedef Real value_type;
	typedef Interpolation<LinearInterpolation, value_type, Real> inter_type;
	typedef MultiDimesionInterpolation<BiLinearInterpolation, value_type> inter2d_type;

	static constexpr int ZAxis = 2;
	static constexpr int XAxis = (ZAxis + 1) % 3;
	static constexpr int YAxis = (ZAxis + 2) % 3;
	enum
	{
		NDIMS = 2
	};
private:
	std::string desc;
//	size_t nw; // Number of horizontal R grid  points
//	size_t nh; // Number of vertical Z grid points
//	Real rdim; // Horizontal dimension in meter of computational box
//	Real zdim; // Vertical dimension in meter of computational box
//	Real rleft; // Minimum R in meter of rectangular computational box
//	Real zmid; // Z of center of computational box in meter
	Real rmaxis = 1.0; // R of magnetic axis in meter
	Real zmaxis = 1.0; // Z of magnetic axis in meter
//	Real simag; // Poloidal flux at magnetic axis in Weber / rad
//	Real sibry; // Poloidal flux at the plasma boundary in Weber / rad
	Real rcentr = 0.5; // R in meter of  vacuum toroidal magnetic field BCENTR
	Real bcentr = 0.5; // Vacuum toroidal magnetic field in Tesla at RCENTR
	Real current = 1.0; // Plasma current in Ampere

	nTuple<NDIMS, size_t> dims_;
	nTuple<NDIMS, Real> rzmin_;
	nTuple<NDIMS, Real> rzmax_;

//	inter_type fpol_; // Poloidal current function in m-T $F=RB_T$ on flux grid
//	inter_type pres_; // Plasma pressure in $nt/m^2$ on uniform flux grid
//	inter_type ffprim_; // $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
//	inter_type pprim_; // $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid

	inter2d_type psirz_; // Poloidal flux in Webber/rad on the rectangular grid points

//	inter_type qpsi_; // q values on uniform flux grid from axis to boundary

	std::vector<nTuple<NDIMS, Real> > rzbbb_; // R,Z of boundary points in meter
	std::vector<nTuple<NDIMS, Real> > rzlim_; // R,Z of surrounding limiter contour in meter

	std::map<std::string, inter_type> profile_;

public:
	GEqdsk()
	{

	}
	GEqdsk(std::string const &fname)
	{
		Read(fname);
	}
	template<typename TDict>
	GEqdsk(TDict const &dict)
	{
		Read(dict["File"].template as<std::string>());
	}

	~GEqdsk()
	{
	}

	void Load(std::string const & fname)
	{
		Read(fname);
	}
	std::string Save(std::string const & path) const;

	void Read(std::string const &fname);

	void Write(std::string const &fname);

	void ReadProfile(std::string const &fname);

	inline value_type Profile(std::string const & name, Real x, Real y) const
	{
		return profile_.at(name)(psi(x, y));
	}

	inline value_type Profile(std::string const & name, Real p) const
	{
		return profile_.at(name)(p);
	}

	std::string const &Description() const
	{
		return desc;
	}
	nTuple<NDIMS, Real> const & GetMin() const
	{
		return rzmin_;
	}

	nTuple<NDIMS, Real> const &GetMax() const
	{
		return rzmax_;
	}

	nTuple<NDIMS, size_t> const &GetDimension() const
	{
		return dims_;
	}

	std::ostream & Print(std::ostream & os);

	inline std::vector<nTuple<NDIMS, Real> > const & Boundary() const
	{
		return rzbbb_;
	}
	inline std::vector<nTuple<NDIMS, Real> > const & Limiter() const
	{
		return rzlim_;
	}

	inline value_type psi(Real x, Real y) const
	{
		return psirz_.eval(x, y);
	}

	inline nTuple<3, Real> B(Real x, Real y, unsigned int VecZAxis = 2) const
	{
		auto gradPsi = psirz_.diff(x, y);

		nTuple<3, Real> res;
		res[(VecZAxis + 1) % 3] = gradPsi[1] / x;
		res[(VecZAxis + 2) % 3] = -gradPsi[0] / x;
		res[(VecZAxis + 3) % 3] = Profile("fpol", x, y) / x;
		return std::move(res);

	}

	inline Real JT(Real x, Real y, unsigned int ToZAxis = 2) const
	{
		return x * Profile("pprim", x, y) + Profile("ffprim", x, y) / x;
	}

	bool CheckProfile(std::string const & name) const
	{
		return (name == "psi") || (name == "JT") || (name == "B") || (profile_.find(name) != profile_.end());
	}

	template<typename TModel>
	void SetUpModel(TModel *model) const;

	template<typename TF>
	void GetProfile(std::string const & name, TF* f) const
	{
		GetProfile_(std::integral_constant<bool, is_nTuple<decltype(get_value(*f,0UL))>::value>(), name, f);
	}

private:

	template<typename TF>
	void GetProfile_(std::integral_constant<bool, true>, std::string const & name, TF* f) const;
	template<typename TF>
	void GetProfile_(std::integral_constant<bool, false>, std::string const & name, TF* f) const;
}
;
template<typename TModel>
void GEqdsk::SetUpModel(TModel *model) const
{

	model->Set(model->SelectByPolylines(VERTEX, Limiter()), model->RegisterMaterial("Vacuum"));

	model->Set(model->SelectByPolylines(VERTEX, Boundary()), model->RegisterMaterial("Plasma"));

}
template<typename TF>
void GEqdsk::GetProfile_(std::integral_constant<bool, true>, std::string const & name, TF* f) const
{
	typedef typename TF::mesh_type mesh_type;
	static constexpr unsigned int IForm = TF::IForm;

	if (name == "B")
	{

		for (auto s : f->GetRange())
		{
			auto x = f->mesh.InvMapTo(f->mesh.GetCoordinates(s), ZAxis);

			get_value(*f, s) = f->mesh.Sample(Int2Type<IForm>(), s, B(x[XAxis], x[YAxis], mesh_type::ZAxis));
		}
	}
	else if (name == "JT")
	{

		for (auto s : f->GetRange())
		{
			auto x = f->mesh.InvMapTo(f->mesh.GetCoordinates(s), ZAxis);

			get_value(*f, s) = f->mesh.Sample(Int2Type<IForm>(), s, JT(x[XAxis], x[YAxis], mesh_type::ZAxis));
		}
	}
	else
	{
		WARNING << "Geqdsk:  Object '" << name << "'[vector]  does not exist!";
	}
	UpdateGhosts(f);
}

template<typename TF>
void GEqdsk::GetProfile_(std::integral_constant<bool, false>, std::string const & name, TF* f) const
{
	typedef typename TF::mesh_type mesh_type;
	static constexpr unsigned int IForm = TF::IForm;

	if (name == "psi")
	{

		for (auto s : f->GetRange())
		{
			auto x = f->mesh.InvMapTo(f->mesh.GetCoordinates(s), ZAxis);

			get_value(*f, s) = psi(x[XAxis], x[YAxis]);
		}
	}
	else if (CheckProfile(name))
	{
		for (auto s : f->GetRange())
		{
			auto x = f->mesh.InvMapTo(f->mesh.GetCoordinates(s), ZAxis);
			get_value(*f, s) = Profile(name, x[XAxis], x[YAxis]);
		}
	}
	else
	{
		WARNING << "Geqdsk:  Object '" << name << "'[scalar]  does not exist!";
	}
	UpdateGhosts(f);
}
std::string XDMFWrite(GEqdsk const & self, std::string const &fname, int flag);

}
// namespace simpla

#endif /* GEQDSK_H_ */
