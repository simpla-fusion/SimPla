Description="For Cold Plasma Dispersion" -- description or other text things.
-- SI Unit System
c 	 	= 299792458  -- m/s
e		= 1.60217656e-19 -- C
me		= 9.10938291e-31 --kg
mp		= 1.672621777e-27 --kg
mp_me	= 1836.15267245 --
KeV 	= 1.1604e7    -- K
Tesla 	= 1.0       -- Tesla
PI		= 3.141592653589793
TWOPI	= PI*2
k_B		= 1.3806488e-23 --Boltzmann_constant
epsilon0	=8.8542e-12
--

k_parallel=6.5
Btor	= 1.0  * Tesla
Ti 		=  0.0003 * KeV
Te 		=  0.0000005 * KeV
N0 		= 1.0e19 -- m^-3


omega_ci = e * Btor/mp -- e/m_p B0 rad/s
vTi		= math.sqrt(k_B*Ti*2/mp)
rhoi 	= vTi/omega_ci    -- m

omega_ce = e * Btor/me -- e/m_p B0 rad/s
vTe		= math.sqrt(k_B*Te*2/me)
rhoe 	= vTe/omega_ce    -- m
omeaga_pe=math.sqrt(N0*e*e/(me*epsilon0))

NX = 128
NY = 1
NZ = 1
LX = 10 --m --100000*rhoi --0.6
LY = 0 --2.0*math.pi/k0
LZ = 0 -- 2.0*math.pi/18
GW = 5

omega_ext=omega_ci*1.9


-- From Gan
InitN0 = function(x)
	local X0 = 12*LX/NX;
	local DEN_JUMP = 0.4*LX;
	local DEN_GRAD = 0.2*LX;
	local AtX0 = 2./math.pi*math.atan((-DEN_JUMP)/DEN_GRAD);
	local AtLX = 2./math.pi*math.atan((LX-DEN_JUMP-X0)/DEN_GRAD);
	local DenCof = 1./(AtLX-AtX0);
	local dens1 = DenCof*(2./math.pi*math.atan((x[0]-DEN_JUMP)/DEN_GRAD)-AtX0);

	return N0 -- *dens1
end

InitB0 = function(x)
	return {0,0,Btor}
end



InitValue = {

	---[[
	E=function(x)

		local res = 0.0;
		for i=1,20 do
			res=res+math.sin(x[0]/LX*TWOPI* i);
		end;

		return {res,res,res}
	end
	--]]


	--	E 	= 0.0
	, J 	= 0.0
	, B 	= InitB0
	, ne 	= InitN0

}



Grid=
{
	Type = "RectMesh",

	UnitSystem={Type="SI"},

	Topology =
	{
		Type="RectMesh",
		Dimensions={NX,NY,NZ}, -- number of grid, now only first dimension is valid

	},
	Geometry=
	{
		Type="Origin_DxDyDz",
		Min={0.0,0.0,0.0},
		Max={LX,LY,LZ},
	},
	dt= 0.1*LX/NX/c -- time step
}

--[[
Model=
{
{Type="Vacuum",Range={{0.2*LX,0,0},{0.8*LX,0,0}},Op="Set"},

{Type="Plasma",
Select=function(x,y,z)
return x>1.0 and x<2.0
end
,Op="Register"},
}
--]]

FieldSolver=
{
--	PML=  {Min={0.1*LX,0.1*LY,0.1*LZ},Max={0.9*LX,0.9*LY,0.9*LZ}}
}





Constraints=
{
	---[[
	{
		DOF="J",
		Select={Type="Range",Value={{0.1*LX,0,0}}},
		Operation= function(t,x,f )
			local tau = t*omega_ext
			local amp=	math.sin(tau) --*(1-math.exp(-tau*tau)
			return { f[0],f[1]+amp,f[2]}
		end
	},
	{
		DOF="E",
		Select={Type="Range",
			Value= function(x)
				return x[0]< 0.05*LX or x[0]>0.95*LX
			end
		},
		Operation= function(t,x,f )
			return {  0, 0, 0}
		end
	},

--]]
}


ParticleConstraints=
{
--	{
--		DOF="ParticlesBoundary",
--		Select={Type="Surface", DistanceToBoundary=0.02*LX},
--		Operation= "Reflecting"
--	},

--	{
--		DOF="J",
--		Select={Type="Range",
--			Value= function(x)
--				return x[0]< 0.05*LX or x[0]>0.95*LX
--			end},
--		Operation= function(t,x,f )
--			return { 0,0,0}
--		end
--	},

--	{
--		DOF="J",
--		Select={Type="Range",
--			Value= function(x)
--				return true
--			end},
--		Operation= function(t,x,f )
--			return { f[0],f[1],0}
--		end
--	},


}

---[[
Particles={
	H 	= {Type="GGauge16",Mass=mp,Charge=e,Temperature=Ti,Density=InitN0,PIC=200,
		EnableImplicit =true,EnableSorting=true,Commands=ParticleConstraints },
--	H 	= {Type="Default",Mass=mp,Charge=e,Temperature=Ti,Density=InitN0,PIC=200,
--		EnableImplicit =true,EnableSorting=true,Commands=ParticleConstraints },
--	ele = {Type="Default",Mass=me,Charge=-e,Temperature=Te,Density=InitN0,PIC=200 ,
--		EnableImplicit =true,EnableSorting=true,Commands=ParticleConstraints },
--	H 	= {Type="DeltaF",Mass=mp,Charge=e,Temperature=Ti,Density=InitN0,PIC=100,
--		EnableImplicit =true,EnableSorting=true,Commands=ParticleConstraints },
--	ele 	= {Type="DeltaF",Mass=me,Charge=-e,Temperature=Te,Density=InitN0,PIC=100 ,
--		EnableImplicit =true,EnableSorting=true,Commands=ParticleConstraints }
--	ele  = {Type="ColdFluid",Mass=me,Charge=-e,Density=InitN0, EnableImplicit=true },
--	H  = {Type="ColdFluid",Mass=mp,Charge=e,Density=InitN0  },
}
--]]


-- The End ---------------------------------------

