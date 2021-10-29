function vq=shearQuaternion(q)

%SHEARQUATERNION   Computes the shear corresponding to a given quaternion 
%   VQ=SHEARQUATERNION(Q)
%   * V is the Shear (Nx8)
%   ** Q is the quaternion (Nx4)
%

x=q(:,2);y=q(:,3);z=q(:,4);w=q(:,1);        
x2=x.^2;y2=y.^2;z2=z.^2;
xy=x.*y;xw=x.*w;yz=y.*z;zw=z.*w;
yzxw=yz-xw;xyzw=xy-zw;
xyzwyzxw=xyzw.*yzxw;

vq(:,1)=(x2+y2)./yzxw;
vq(:,2)=(y2-z2)./xyzw-2*(yz.*(x2+y2))./xyzwyzxw;
vq(:,3)=2*yz./xyzw;
vq(:,4)=-2*yzxw;
vq(:,5)=2*xyzw;
vq(:,6)=-2*xy./yzxw;
vq(:,7)=(x2-y2)./yzxw+2*(xy.*(y2+z2))./xyzwyzxw;
vq(:,8)=-(y2+z2)./xyzw;
