Description="RF Wave in Tokamak" -- description or other text things.
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
Btor	=  3.0  * Tesla
Ti 		=  0.5 * KeV
Te 		=  0.5 * KeV
N0 		= 4.0e19 -- m^-3


omega_ci = e * Btor/mp -- e/m_p B0 rad/s
vTi		= math.sqrt(k_B*Ti*2/mp)
rhoi 	= vTi/omega_ci    -- m
omega_pi=math.sqrt(N0*e*e/(mp*epsilon0))

omega_ce = e * Btor/me -- e/m_p B0 rad/s
vTe		= math.sqrt(k_B*Te*2/me)
rhoe 	= vTe/omega_ce    -- m
omega_pe=math.sqrt(N0*e*e/(me*epsilon0))

omega_lhw=1.0/math.sqrt(1.0/(omega_pi*omega_pi)+1.0/(omega_ce*omega_ci))

NX = 128
NY = 128
NZ = 1
LX = 10 --  --100000*rhoi --0.6
LY = 20 -- 2.0*math.pi/k0
LZ = 30 -- 2.0*math.pi/18
GW = 5




omega_ext=1.9*omega_ci

--print( "Angle Frequency of antenna =",omega_ext )


Model=
{

	--Type = "ExplicitEMContext_Cartesian_UniformArray",

	Type ="ExplicitEMContext_Cylindrical2_UniformArray",

	UnitSystem={Type="SI"},

	GFile='/home/salmon/workspace/SimPla/example/gfile/g038300.03900',

	Mesh={


		Min={1.2,-1.2,0.0 },

		Max={2.4,1.2,TWOPI/2  },

		Dimensions={NX,NY,NZ}, -- number of grid, now only first dimension is valid

		CFL =0.4,

	},

}


--
--FieldSolver=
--{
--	PML=  {Min={0.1*LX,0.1*LY,0.1*LZ},Max={0.9*LX,0.9*LY,0.9*LZ}}
--}



antenna_pos=2.25
Constraints=
{

	{
		DOF="J",
		Select=function(x)
			return x[0]>antenna_pos-0.01 and x[0]<antenna_pos+0.01 and x[1]>-0.1 and x[1]<0.1
		end,
		Operation= function(t,x,f )
			local tau = t*omega_ext+ x[2]*TWOPI/(Model.Mesh.Max[3]-Model.Mesh.Min[3])
			local amp=	math.sin(tau) --*(1-math.exp(-tau*tau)

			return { f[0],f[1],f[2]+amp}
		end
	},
--
--	{
--		DOF="E",
--		Select={Type="Boundary",In="Vacuum"},
--		Operation= function(t,x,f )	 return { 0,  0, 0} end
--	},
--	{
--		DOF="B",
--		Select={Type="Boundary",In="Vacuum"},
--		Operation= function(t,x,f ) return {  0,  0, 0}	end
--	},
--	{
--		DOF="E",
--		Select={ Material="NONE"},
--		Operation= function(t,x,f )	 return { 0,0,0} end
--	},
--	{
--		DOF="B",
--		Select={ Material="NONE"},
--		Operation= function(t,x,f ) return { 0,0,0}	end
--	},

}
--[[

ParticleConstraints=
{
{
DOF="ParticlesBoundary",
Select={Type="Surface", DistanceToBoundary=0.02*LX},
Operation= "Reflecting"
},

{
DOF="J",
Select={Type="Range",
Value= function(x)
return x[0]< 0.05*LX or x[0]>0.95*LX
end},
Operation= function(t,x,f )
return { 0,0,0}
end
},

{
DOF="J",
Select={Type="Range",
Value= function(x)
return true
end},
Operation= function(t,x,f )
return { f[0],f[1],0}
end
},


}
--]]

Particles={
	--	H 		= {Type="DeltaF",		Mass=mp,Charge=e,	Temperature=Ti,	Density=N0,	PIC=200 },
	--	H  		= {Type="Implicit",		Mass=mp,Charge=e,	Temperature=Ti,	Density=N0,	PIC=200	,ScatterN=true},
	  H 		= {Type="FullF",		Mass=mp,Charge=e,	Temperature=Ti,	Density=N0, PIC=40,Select={Material="Plasma"},
	       ScatterN=true,DumpParticle=false,DumpParticle=true},
	--H    	= {Type="ColdFluid",	Mass=mp,Charge=e,	Select={Material="Plasma"} ,Ratio=0.1},


	--	ele 	= {Type="Default",	 Mass=me, Charge=-e,	Density=N0, Temperature=Te,	PIC=200 },
	--	ele 	= {Type="DeltaF",	 Mass=me, Charge=-e,	Density=N0, Temperature=Te,	PIC=200 },
--		ele 	= {Type="Implicit",	 Mass=me, Charge=-e,	Density=N0, Temperature=Te, PIC=200,ScatterN=true},
	--ele 	= {Type="ColdFluid", Mass=me, Charge=-e,	Select={Material="Plasma"},Ratio=0.1 },
}



-- The End ---------------------------------------

