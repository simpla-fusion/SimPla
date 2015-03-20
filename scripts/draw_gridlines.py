fig,ax=subplots()

ax.set_xticks(arange(0.5,8.5,1),minor=True)
ax.set_xticks(arange(0,8,1),minor=False)
ax.xaxis.grid(True,which='minor',color='r',linestyle='--')
ax.xaxis.grid(True,which='major',color='b',linestyle='-')
 
ax.set_yticks(arange(0.5,8.5,1),minor=True)
ax.set_yticks(arange(0,8,1),minor=False)
ax.yaxis.grid(True,which='major',color='b',linestyle='-')
ax.yaxis.grid(True,which='minor',color='r',linestyle='--')

 plot(f1["/p0"][:][:,0],f1["/p0"][:][:,1])
