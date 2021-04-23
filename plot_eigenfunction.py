import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

with h5py.File('eigenfunction.h5','r') as f:
    UltraS = f['UltraS'][:,0] + 1.j*f['UltraS'][:,1]
    Physical = f['Physical'][:,0] + 1.j*f['Physical'][:,1]
    UltraSl = f['UltraSl'][:,0] + 1.j*f['UltraSl'][:,1]
    Physicall = f['Physicall'][:,0] + 1.j*f['Physicall'][:,1]
    y = f['y'][:,0]
ny=y.shape[0]
norm1 = UltraS[np.argmax(np.abs(UltraS[:ny]))]
norm2 = Physical[np.argmax(np.abs(Physical[:ny]))]
norm1l = UltraSl[np.argmax(np.abs(UltraSl[:ny]))]
norm2l = Physicall[np.argmax(np.abs(Physicall[:ny]))]
u1,v1,w1,p1,au1,av1,aw1,ap1 = UltraS.reshape(8,ny)/norm1
u2,v2,w2,p2,au2,av2,aw2,ap2 = Physical.reshape(8,ny)/norm2
u1l,v1l,w1l,p1l,au1l,av1l,aw1l,ap1l = UltraSl.reshape(8,ny)/norm1l
u2l,v2l,w2l,p2l,au2l,av2l,aw2l,ap2l = Physicall.reshape(8,ny)/norm2l

fig,ax = plt.subplots(figsize=(8,8),nrows=2,ncols=4,tight_layout=True)
ax[0,0].plot(np.abs(u1),y,'-',label='UltraS')
ax[0,0].plot(u1.real,y,'-',label='real(UltraS)')
ax[0,0].plot(u1.imag,y,'-',label='imag(UltraS)')
ax[0,0].plot(np.abs(u2),y,'--',label='Physical')
ax[0,0].plot(u2.real,y,'--',label='real(Physical)')
ax[0,0].plot(u2.imag,y,'--',label='imag(Physical)')
ax[0,1].plot(np.abs(v1),y,'-',label='UltraS')
ax[0,1].plot(v1.real,y,'-',label='real(UltraS)')
ax[0,1].plot(v1.imag,y,'-',label='imag(UltraS)')
ax[0,1].plot(np.abs(v2),y,'--',label='Physical')
ax[0,1].plot(v2.real,y,'--',label='real(Physical)')
ax[0,1].plot(v2.imag,y,'--',label='imag(Physical)')
ax[0,2].plot(np.abs(w1),y,'-',label='UltraS')
ax[0,2].plot(w1.real,y,'-',label='real(UltraS)')
ax[0,2].plot(w1.imag,y,'-',label='imag(UltraS)')
ax[0,2].plot(np.abs(w2),y,'--',label='Physical')
ax[0,2].plot(w2.real,y,'--',label='real(Physical)')
ax[0,2].plot(w2.imag,y,'--',label='imag(Physical)')
ax[0,3].plot(np.abs(p1),y,'-',label='UltraS')
ax[0,3].plot(p1.real,y,'-',label='real(UltraS)')
ax[0,3].plot(p1.imag,y,'-',label='imag(UltraS)')
ax[0,3].plot(np.abs(p2),y,'--',label='Physical')
ax[0,3].plot(p2.real,y,'--',label='real(Physical)')
ax[0,3].plot(p2.imag,y,'--',label='imag(Physical)')
ax[1,0].plot(np.abs(au1),y,'-',label='UltraS')
ax[1,0].plot(au1.real,y,'-',label='real(UltraS)')
ax[1,0].plot(au1.imag,y,'-',label='imag(UltraS)')
ax[1,0].plot(np.abs(au2),y,'--',label='Physical')
ax[1,0].plot(au2.real,y,'--',label='real(Physical)')
ax[1,0].plot(au2.imag,y,'--',label='imag(Physical)')
ax[1,1].plot(np.abs(av1),y,'-',label='UltraS')
ax[1,1].plot(av1.real,y,'-',label='real(UltraS)')
ax[1,1].plot(av1.imag,y,'-',label='imag(UltraS)')
ax[1,1].plot(np.abs(av2),y,'--',label='Physical')
ax[1,1].plot(av2.real,y,'--',label='real(Physical)')
ax[1,1].plot(av2.imag,y,'--',label='imag(Physical)')
ax[1,2].plot(np.abs(aw1),y,'-',label='UltraS')
ax[1,2].plot(aw1.real,y,'-',label='real(UltraS)')
ax[1,2].plot(aw1.imag,y,'-',label='imag(UltraS)')
ax[1,2].plot(np.abs(aw2),y,'--',label='Physical')
ax[1,2].plot(aw2.real,y,'--',label='real(Physical)')
ax[1,2].plot(aw2.imag,y,'--',label='imag(Physical)')
ax[1,3].plot(np.abs(ap1),y,'-',label='UltraS')
ax[1,3].plot(ap1.real,y,'-',label='real(UltraS)')
ax[1,3].plot(ap1.imag,y,'-',label='imag(UltraS)')
ax[1,3].plot(np.abs(ap2),y,'--',label='Physical')
ax[1,3].plot(ap2.real,y,'--',label='real(Physical)')
ax[1,3].plot(ap2.imag,y,'--',label='imag(Physical)')
ax[0,0].legend(loc='best',numpoints=1)
fig.show()

fig,ax = plt.subplots(figsize=(8,8),nrows=2,ncols=4,tight_layout=True)
ax[0,0].plot(np.abs(u1l),y,'-',label='UltraSl')
ax[0,0].plot(u1l.real,y,'-',label='real(UltraSl)')
ax[0,0].plot(u1l.imag,y,'-',label='imag(UltraSl)')
ax[0,0].plot(np.abs(u2l),y,'--',label='Physicall')
ax[0,0].plot(u2l.real,y,'--',label='real(Physicall)')
ax[0,0].plot(u2l.imag,y,'--',label='imag(Physicall)')
ax[0,1].plot(np.abs(v1l),y,'-',label='UltraSl')
ax[0,1].plot(v1l.real,y,'-',label='real(UltraSl)')
ax[0,1].plot(v1l.imag,y,'-',label='imag(UltraSl)')
ax[0,1].plot(np.abs(v2l),y,'--',label='Physicall')
ax[0,1].plot(v2l.real,y,'--',label='real(Physicall)')
ax[0,1].plot(v2l.imag,y,'--',label='imag(Physicall)')
ax[0,2].plot(np.abs(w1l),y,'-',label='UltraSl')
ax[0,2].plot(w1l.real,y,'-',label='real(UltraSl)')
ax[0,2].plot(w1l.imag,y,'-',label='imag(UltraSl)')
ax[0,2].plot(np.abs(w2l),y,'--',label='Physicall')
ax[0,2].plot(w2l.real,y,'--',label='real(Physicall)')
ax[0,2].plot(w2l.imag,y,'--',label='imag(Physicall)')
ax[0,3].plot(np.abs(p1l),y,'-',label='UltraSl')
ax[0,3].plot(p1l.real,y,'-',label='real(UltraSl)')
ax[0,3].plot(p1l.imag,y,'-',label='imag(UltraSl)')
ax[0,3].plot(np.abs(p2l),y,'--',label='Physicall')
ax[0,3].plot(p2l.real,y,'--',label='real(Physicall)')
ax[0,3].plot(p2l.imag,y,'--',label='imag(Physicall)')
ax[1,0].plot(np.abs(au1l),y,'-',label='UltraSl')
ax[1,0].plot(au1l.real,y,'-',label='real(UltraSl)')
ax[1,0].plot(au1l.imag,y,'-',label='imag(UltraSl)')
ax[1,0].plot(np.abs(au2l),y,'--',label='Physicall')
ax[1,0].plot(au2l.real,y,'--',label='real(Physicall)')
ax[1,0].plot(au2l.imag,y,'--',label='imag(Physicall)')
ax[1,1].plot(np.abs(av1l),y,'-',label='UltraSl')
ax[1,1].plot(av1l.real,y,'-',label='real(UltraSl)')
ax[1,1].plot(av1l.imag,y,'-',label='imag(UltraSl)')
ax[1,1].plot(np.abs(av2l),y,'--',label='Physicall')
ax[1,1].plot(av2l.real,y,'--',label='real(Physicall)')
ax[1,1].plot(av2l.imag,y,'--',label='imag(Physicall)')
ax[1,2].plot(np.abs(aw1l),y,'-',label='UltraSl')
ax[1,2].plot(aw1l.real,y,'-',label='real(UltraSl)')
ax[1,2].plot(aw1l.imag,y,'-',label='imag(UltraSl)')
ax[1,2].plot(np.abs(aw2l),y,'--',label='Physicall')
ax[1,2].plot(aw2l.real,y,'--',label='real(Physicall)')
ax[1,2].plot(aw2l.imag,y,'--',label='imag(Physicall)')
ax[1,3].plot(np.abs(ap1l),y,'-',label='UltraSl')
ax[1,3].plot(ap1l.real,y,'-',label='real(UltraSl)')
ax[1,3].plot(ap1l.imag,y,'-',label='imag(UltraSl)')
ax[1,3].plot(np.abs(ap2l),y,'--',label='Physicall')
ax[1,3].plot(ap2l.real,y,'--',label='real(Physicall)')
ax[1,3].plot(ap2l.imag,y,'--',label='imag(Physicall)')
ax[0,0].legend(loc='best',numpoints=1)
fig.show()

# show inner products
print(UltraS@(UltraS.conj()))
print(Physical@(Physical.conj()))
print(UltraSl@(UltraSl.conj()))
print(Physicall@(Physicall.conj()))

Re = 1000.0/1.7208
i = 1.j
O = np.zeros((ny,ny))
O4 = np.zeros((4*ny,4*ny))
I = np.eye(ny)
I4 = np.eye(4*ny)
L2 = np.block([
    [-I, O, O, O],
    [ O,-I, O, O],
    [ O, O,-I, O],
    [ O, O, O, O],
    ])
M = np.block([
    [I4, O4],
    [O4,-L2]
    ])
dLdomega4 = np.block([
    [i*Re*I, O, O, O],
    [O, i*Re*I, O, O],
    [O, O, i*Re*I, O],
    [O, O,      O, O],
    ])
dLdomega = np.block([
    [O4, O4],
    [dLdomega4, O4]
    ])

def inn(a,b):
    return (a)@(b.conj())

print('UltraS cg = ',inn(UltraS,M@UltraSl)/inn(dLdomega@UltraS,UltraSl))
print('Physical cg = ',inn(Physical,M@Physicall)/inn(dLdomega@Physical,Physicall))
print(inn(UltraS,UltraSl))
print(inn(UltraS,M@UltraSl))
print(inn(M@UltraS,UltraSl))
print(inn(Physical,Physicall))
print(inn(Physical,M@Physicall))
print(inn(M@Physical,Physicall))
tmp = inn(UltraS,M@UltraSl)
UltraS2 = UltraS/tmp
tmp = inn(Physical,M@Physicall)
Physical2 = Physical/tmp
print(inn(UltraS2,M@UltraSl))
print(inn(Physical2,M@Physicall))
print('UltraS2 cg = ',inn(UltraS2,M@UltraSl)/inn(dLdomega@UltraS2,UltraSl))
print('UltraS2 lchange cg = ',inn(UltraS2,M@Physicall)/inn(dLdomega@UltraS2,Physicall))
print('Physical2 cg = ',inn(Physical2,M@Physicall)/inn(dLdomega@Physical2,Physicall))

tmp1 = UltraS/norm1
tmp2 = Physical/norm2
tmp1l = UltraSl/norm1l
tmp2l = Physicall/norm2l

print('error = ',np.sum(np.abs(tmp1-tmp2)))
print('errorl = ',np.sum(np.abs(tmp1l-tmp2l)))


input('Enter to Exit')
