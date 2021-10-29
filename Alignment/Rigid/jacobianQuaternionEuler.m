function Je=jacobianQuaternionEuler(e)

%JACOBIANSHEARQUATERNION   Computes the Jacobian of the shear parameters 
%with respect to the quaternion 
%   JE=JACOBIANSHEARQUATERNION(E)
%   * JE is the Jacobian (4x3)
%   ** E are the Euler angles (ZYX)
%

c=cos(e/2);c1=c(3);c2=c(2);c3=c(1);
s=sin(e/2);s1=s(3);s2=s(2);s3=s(1);
ssc=s1.*s2.*c3;
ccs=c1.*c2.*s3;
scs=s1.*c2.*s3;
csc=c1.*s2.*c3;
css=c1.*s2.*s3;
scc=s1.*c2.*c3;
sss=s1.*s2.*s3;
ccc=c1.*c2.*c3;
%Je=0.5*[  ssc-ccs    scs-csc    css-scc;
%        -(csc+scs) -(ssc+ccs)   ccc+sss;
%          scc-css    ccc-sss    ccs-ssc;
%          ccc+sss  -(css+scc) -(csc+scs)];%This is what appears in "A 
       %tutorial on SE(3) transformation parameterizations and on-manifold 
       %optimization,", JL Blanco Claraco, Universidad de Malaga, 2019
      
Je=0.5*[-(ccs+ssc) -(csc+scs) -(scc+css);
         -scs+csc   -ssc+ccs    ccc-sss;
        -(css+scc)   ccc+sss  -(ssc+ccs);
          ccc-sss   -css+scc   -scs+csc];%This is what works for our 
      %convention
