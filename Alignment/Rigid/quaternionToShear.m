function [vshear,tshear,qext]=quaternionToShear(q,typ)

%QUATERNIONTOSHEAR   Converts quaternions+translation to shears following 
%[1] JS Welling, WF Eddy, TK Young, "Rotation of 3D volumes by 
%Fourier-interpolated shears," Graph Mod 68:356-370, 2006
%   [VSHEAR,TSHEAR,QEXT]=QUATERNIONTOSHEAR(Q)
%   * Q is a quaternion+translation representation (w,x,y,z,tx,ty,tz)
%   * TYP is the criterion used to choose the shear, one of the following,
%   'eucl', 'manh', 'maxi', 'volu', defaults to 'maxi'
%   ** VSHEAR is the vector of shears as (a,b,c,d,e,f,g,h,tx',ty',tz')
%   ** TSHEAR is the shear type, 1 for SySzSxSy, 2 for SzSxSySz, 3 for
%   SxSySzSx
%   ** QEXT is an extended quaternion of size 6x4 (w,x,y,z)
%

if nargin<2 || isempty(typ);typ='maxi';end%We choose maxi as it may be good to avoid boundary effects

q(end+1:7)=0;q=q(1:7);

t=q(5:7);
q=double(q(1:4));
q0=q;
st=0;
while ~st
    q=q0+1e-12*randn(1,4);%To avoid singularities
    q=q/norm(q);
    q=[q([1 2 3 4]);q([1 3 4 2]);q([1 4 2 3])];
    q=[q;q];q(4:6,1)=-q(4:6,1);
    qext=q;
    vshear=shearQuaternion(q);    
    if ~any(isnan(vshear(:)));break;end
end
if strcmp(typ,'eucl');[~,tshear]=min(sum(vshear.^2,2));   
elseif strcmp(typ,'manh');[~,tshear]=min(sum(abs(vshear),2));
elseif strcmp(typ,'maxi');[~,tshear]=min(max(abs(vshear),[],2));
elseif strcmp(typ,'volu');[~,tshear]=min(abs(sum(vshear(:,[1 7]),2))+abs(sum(vshear(:,[2 8]),2))+sum(abs(vshear(:,3:6)),2));
else error('Not contemplated %s criterion for shear selection',typ);
end
vshear=vshear(tshear,:);

t=circshift(t,-(tshear-1),2);
if tshear<=3;t=t-[0 vshear(1)*t(3)+vshear(2)*t(1) vshear(3)*t(1)];else t=t+[vshear(6)*t(3) vshear(7)*t(3)+vshear(8)*t(1) 0];end
t=circshift(t,(tshear-1),2);
vshear(9:11)=t;
