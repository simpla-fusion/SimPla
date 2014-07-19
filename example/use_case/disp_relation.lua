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
Ti 		=  0.0000005 * KeV
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
LX = 1  --m --100000*rhoi --0.6
LY = 20 --2.0*math.pi/k0
LZ = 30 -- 2.0*math.pi/18

InitValue = {

	---[[
	E=function(x)

		local res = 0.0;
		for i=1,1 do
			res=res+math.sin(x[0]/LX*TWOPI* i + x[1]/LY*TWOPI);
		end;

		return {res,res,res}
	end
	--]]
	, J 	= 0.0
	, B 	=  {0,0,0.0}
}

Model=
{

	Type = "ExplicitEMContext_Cartesian_UniformArray",

	--Type ="ExplicitEMContext_Cylindrical2_UniformArray",

	UnitSystem={Type="SI"},

	Mesh={

		Min={0,0,0 },

		Max={LX,LY,LZ},

		Dimensions={NX,NY,1},

		CFL =0.5,

	},

}




Particles={
	H 	= {Type="Default",	Mass=mp,Charge=e,	Temperature=Ti,	Density=N0,	PIC=200},
--	ele = {Type="Implicit",	Mass=me,Charge=-e,	Temperature=Te,	Density=N0,	PIC=20},
--	H 	= {Type="DeltaF",Mass=mp,Charge=e,Temperature=Ti,Density=N0,PIC=200  },

--	ele1= {Type="DeltaF",Mass=me,Charge=-e,Temperature=Te,Density=InitN0,PIC=100 },
--	ele = {Type="ColdFluid",Mass=me,Charge=-e,Density=N0 },
--	H   = {Type="ColdFluid",Mass=mp,Charge=e,Density=N0 },
}


-- The End ---------------------------------------

