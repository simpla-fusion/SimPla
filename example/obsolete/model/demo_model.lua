
NX = 10
NY = 10 
NZ = 1
LX = 10   --m --100000*rhoi --0.6
LY = 10   --2.0*math.pi/k0
LZ = 0   -- 2.0*math.pi/18
GW = 5

Mesh={

    Dimensions={NX ,NY ,NZ },

    Box={{0.0,0.0,0.0},{LX,LY,LZ}}
}
Object=function(v)
   --res=( (v[1]-5.0)* (v[1]-5.0)+ (v[2]-5.0)* (v[2]-5.0))-10
   d1=( (v[1]-LX/2)* (v[1]- LX/2)+ (v[2]-LY/2)* (v[2]-LY/2))-LY*LY*0.04
--   d2=math.max( math.abs(v[1]-LX*0.6)-2.2 , math.abs(v[2]-LY*0.6)-2.2)
--   return math.min(d1,d2)
  return d1
 end
SelectTag = 6

SelectIForm=2

Polylines={
  { 0.0 , 1 ,0 },
  { 0.0 , 9.2 ,0 },
  { 8.3  , 7 ,0 },
  { 8.3  , 4 ,0 },
  { 4  , 2.2 ,0 }
}
