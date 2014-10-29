Description="RF Wave Dispersion Realtion" -- description or other text things.
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
Ti 		=  0.5 * KeV
Te 		=  0.05 * KeV
N0 		= 1.0e18 -- m^-3


omega_ci = e * Btor/mp -- e/m_p B0 rad/s
vTi		= math.sqrt(k_B*Ti*2/mp)
rhoi 	= vTi/omega_ci    -- m
omega_pi=math.sqrt(N0*e*e/(mp*epsilon0))

omega_ce = e * Btor/me -- e/m_p B0 rad/s
vTe		= math.sqrt(k_B*Te*2/me)
rhoe 	= vTe/omega_ce    -- m
omega_pe=math.sqrt(N0*e*e/(me*epsilon0))

omega_lhw=1.0/math.sqrt(1.0/(omega_pi*omega_pi)+1.0/(omega_ce*omega_ci))

NX = 64
NY = 32
NZ = 1
LX = 2   --m --100000*rhoi --0.6
LY = 2   --2.0*math.pi/k0
LZ = 3   -- 2.0*math.pi/18

InitValue = {

--	E=function(x)
--
--		local res = 0.0;
--		for i=1,80  do
--			res=res+math.sin((x[0]-1.0)/LX*TWOPI* i );
--		end;
--
--		return {res,res,res}
--	end


}

Model=
{

	Type = "ExplicitEMContext_Cartesian_UniformArray",

	--	Type ="ExplicitEMContext_Cylindrical2_UniformArray_kz",

	UnitSystem={Type="SI"},

	Mesh={

		Min={0.0,0,0 },

		Max={ LX,LY,TWOPI/100},

		Dimensions={NX,NY,NZ},

		CFL =0.1,

	},

}




Particles={
	H 		= {Type="FullF",		Mass=mp,Charge=e,	Temperature=Ti,
		Density = function(x)
			return math.sin((x[0] )/LX*TWOPI  )
		end
		, PIC=200	,
		ScatterN=true,		DumpParticle=false	},
--	H  		= {Type="Implicit",		Mass=mp,Charge=e,	Temperature=Ti,	Density=N0,	PIC=200	,ScatterN=true},
--  H 		= {Type="DeltaF",		Mass=mp,Charge=e,	Temperature=Ti,	Density=N0, PIC=200 },
--	H    	= {Type="ColdFluid",	Mass=mp,Charge=e,	Density=N0 },


--	ele 	= {Type="Default",	 Mass=me, Charge=-e,	Density=N0, Temperature=Te,	PIC=200 },
--	ele 	= {Type="DeltaF",	 Mass=me, Charge=-e,	Density=N0, Temperature=Te,	PIC=200 },
--	ele 	= {Type="Implicit",	 Mass=me, Charge=-e,	Density=N0, Temperature=Te, PIC=200,ScatterN=true },
--	ele 	= {Type="ColdFluid", Mass=me, Charge=-e,	Density=N0 },
}


-- The End ---------------------------------------

