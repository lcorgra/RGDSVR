function Jq=jacobianShearQuaternion(q)

%JACOBIANSHEARQUATERNION   Computes the Jacobian of the shear parameters 
%with respect to the quaternion 
%   JQ=JACOBIANSHEARQUATERNION(Q)
%   * JQ is the Jacobian (8x4)
%   ** Q is the quaternion
%

x=q(2);y=q(3);z=q(4);w=q(1);
x2=x.^2;y2=y.^2;z2=z.^2;
x2py2=x2+y2;y2mz2=y2-z2;x2my2=x2-y2;y2pz2=y2+z2;
xy=x.*y;xw=x.*w;yz=y.*z;zw=z.*w;
xyz=xy.*z;yzw=yz.*w;xyw=xy.*w;yz2=y.*z2;y2z=y2.*z;xy2=x.*y2;x2y=x2.*y;
yzxw=yz-xw;xyzw=xy-zw;
yzxw2=yzxw.^2;xyzw2=xyzw.^2;
xyzw2yzxw=xyzw2.*yzxw;xyzwyzxw2=xyzw.*yzxw.^2;xyzwyzxw=xyzw.*yzxw;

Jq=[x.*x2py2./yzxw2 2*x./yzxw+w.*x2py2./yzxw2 2*y./yzxw-z.*x2py2./yzxw2 -y.*x2py2./yzxw2;
z.*y2mz2./xyzw2-2*yz2.*x2py2./xyzw2yzxw-2*xyz.*x2py2./xyzwyzxw2 -y.*y2mz2./xyzw2-4*xyz./xyzwyzxw+2*y2z.*x2py2./xyzw2yzxw-2*yzw.*x2py2./xyzwyzxw2 2*y./xyzw-x.*y2mz2./xyzw2-4*y2z./xyzwyzxw-2*z.*x2py2./xyzwyzxw+2*xyz.*x2py2./xyzw2yzxw+2*yz2.*x2py2./xyzwyzxw2 -2*z./xyzw+w.*y2mz2./xyzw2-2*y.*x2py2./xyzwyzxw-2*yzw.*x2py2./xyzw2yzxw+2*y2z.*x2py2./xyzwyzxw2;
2*yz2./xyzw2 -2*y2z./xyzw2 2*z./xyzw-2*xyz./xyzw2 2*y./xyzw+2*yzw./xyzw2;
2*x 2*w -2*z -2*y;
-2*z 2*y 2*x -2*w;
-2*x2y./yzxw2 -2*y./yzxw-2*xyw./yzxw2 -2*x./yzxw+2*xyz./yzxw2 2*xy2./yzxw2;
x.*x2my2./yzxw2+2*xyz.*y2pz2./xyzw2yzxw+2*x2y.*y2pz2./xyzwyzxw2 2*x./yzxw+w.*x2my2./yzxw2+2*y.*y2pz2./xyzwyzxw-2*xy2.*y2pz2./xyzw2yzxw+2*xyw.*y2pz2./xyzwyzxw2 -2*y./yzxw-z.*x2my2./yzxw2+4*xy2./xyzwyzxw+2*x.*y2pz2./xyzwyzxw-2*x2y.*y2pz2./xyzw2yzxw-2*xyz.*y2pz2./xyzwyzxw2 -y.*x2my2./yzxw2+4*xyz./xyzwyzxw+2*xyw.*y2pz2./xyzw2yzxw-2*xy2.*y2pz2./xyzwyzxw2;
-z.*y2pz2./xyzw2 y.*y2pz2./xyzw2 -2*y./xyzw+x.*y2pz2./xyzw2 -2*z./xyzw-w.*y2pz2./xyzw2];
