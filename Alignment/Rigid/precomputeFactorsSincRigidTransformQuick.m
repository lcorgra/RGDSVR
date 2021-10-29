function [et,etg,eth]=precomputeFactorsSincRigidTransformQuick(kGrid,rGrid,T,di,cg,gr,sh,cGrid)

%PRECOMPUTEFACTORSSINCRIGIDTRANSFORMQUICK precomputes the k-space phase 
%multiplicative factors required to apply a rigid transform based on sinc 
%interpolation using 4 shears as described in [1] JS Welling, WF Eddy, 
%TK Young, "Rotation of 3D volumes by Fourier-interpolated shears," Graph 
%Mod 68:356-370, 2006
%   [ET,ETG]=PRECOMPUTEFACTORSSINCRIGIDTRANSFORMQUICK(KGRID,RGRID,T,{DI},{CG},{GR},{SH},{CGRID}) 
%   * KGRID is a grid of points in the spectral domain
%   * RGRID is a grid of points in the spatial domain
%   * T are the parameters of the transform
%   * {DI} is a flag to indicate whether to perform direct or inverse 
%   transform (defaults to 1)
%   * {CG} is a flag to indicate whether to calculate the derivative terms 
%   (defaults to 0). This parameter is inherited from non-quick version,
%   but not used
%   * {GR} is a flag to indicate whether this factors are for the grouped 
%   transform (defaults to 0). This parameter is inherited from non-quick
%   version but not used
%   * {SH} is a flag to indicate whether the grids have already been 
%   ifftshifted (defaults to 0)
%   * {CGRID} is the center of the grid
%   ** ET are the parameters to apply the transform
%   ** ETG are the factors to apply the derivative of the transform 
%   ** ETH are the factors to apply the Hessian of the transform 
%

if nargout==1 && all(T(:)==0);et=[];return;end

if nargin<4 || isempty(di);di=1;end
if nargin<5 || isempty(cg);cg=0;end
if nargin<6 || isempty(gr);gr=1;end%Just for compatibility with precomputeFactorsSincRigidTransform
if nargin<7 || isempty(sh);sh=0;end

N=zeros(1,length(kGrid));
for n=1:length(kGrid);N(n)=numel(kGrid{n});end
if nargin<8 || isempty(cGrid);cGrid=(N/2)+1;end
gpu=isa(kGrid{1},'gpuArray');

%TRANSLATION AND EULER ROTATION ANGLES
et=cell(1,5);
ndT=ndims(T);
tr0=dynInd(T,1:3,ndT);
theta=wrapToPi(dynInd(T,4:6,ndT));

%QUATERNION
%M=eultorotm(flip(theta(:)'),'XYZ')';
%R=generatePrincipalAxesRotations(-1);%4 possible permutations
%M=matfun(@transpose,matfun(@mtimes,matfun(@transpose,M),R));
%q=rotmtoquat(gather(matfun(@transpose,M)));

%M=eultorotm(flip(theta(:)'),'XYZ');
%R=generatePrincipalAxesRotations(-1);%4 possible permutations
%M=matfun(@mtimes,M,R);
%q=rotmtoquat(gather(M));

%We try to accelerate
M=eultorotm(flip(theta(:)'),'XYZ');
R=cat(3,[1 1 1],[-1 -1 1],[-1 1 -1],[1 -1 -1]);
M=bsxfun(@times,M,R);
q=rotmtoquat(gather(M));

qn=abs(q(:,1));
[~,mP]=max(qn);
q=q(mP,:);
R=dynInd(R,mP,3);
et{5}=diag(R)<0;
for n=1:3;et{4}{n}{n}=cGrid(n);end

%SHEARS
%fprintf('Rotation angle %.2fdeg\n',convertRotation(2*acos(q(1)),'rad','deg'))
[vshear,tshear,qext]=quaternionToShear(cat(2,q,tr0(:)'));
tr=[2 1 3 2;
    3 2 1 3;
    1 3 2 1;
    2 3 1 2;
    3 1 2 3;
    1 2 3 1];%Set of shears
qext=qext(tshear,:);
tr=tr(tshear,:);
trd=1+mod([tr;tr+1],3);
vsh=vshear(1:8);
vsh=reshape(vsh,[2 4]);
if tshear<=3;vsh=flip(vsh,2);else vsh=-vsh;end
tsh=vshear(9:11);

%PHASE SHIFTS
mult=-1i;
if di==0;mult=1i;end
mV=1:4;
if ~sh
    for m=1:3;kGrid{m}=ifftshift(kGrid{m},m);end
end
etg=cell(4,3);
for m=mV   
    if m>1;sh=tsh(tr(m));else sh=0;end
    sh=bsxfun(@plus,sh+vsh(1,m)*rGrid{trd(1,m)},vsh(2,m)*rGrid{trd(2,m)});   
    et{1}{m}=exp(bsxfun(@times,mult*kGrid{tr(m)},sh));  
    if cg
        for n=1:2;etg{m}{n}=mult*bsxfun(@times,kGrid{tr(m)},rGrid{trd(n,m)});end
        if m>1;etg{m}{3}=mult*kGrid{tr(m)};end
    end
end
et{2}=tshear;
perq=[1 2 3 4;1 3 4 2;1 4 2 3];
if cg
    etg{5}=jacobianQuaternionEuler(flip(quattoeul(q,'XYZ')));    
    etg{5}=etg{5}(perq(mod(tshear-1,3)+1,:),:);
    if tshear>3;etg{5}(1,:)=-etg{5}(1,:);end
    etg{5}=jacobianShearQuaternion(qext)*etg{5};
    etg{5}=reshape(etg{5},[2 4 3]);
    if tshear<=3;etg{5}=flip(etg{5},2);else etg{5}=-etg{5};end
    etg{5}=reshape(etg{5},[8 3]);
    tg=zeros(3,3);    
    tg(tr(4),trd(2,4))=-vsh(2,4);
    tg(tr(4),trd(1,4))=-vsh(1,4);
    if tshear<=3;tg(tr(3),trd(1,3))=-vsh(1,3);else tg(tr(3),trd(2,3))=-vsh(2,3);end
    tg=eye(3)+tg;
    etp=etg{5};
    etg{5}=blkdiag(tg,etg{5});
    perm=[4 5 6 7 tr(2) 8 9 tr(3) 10 11 tr(4)];
    etg{5}=etg{5}(perm,:);    
    if tshear<=3;etg{5}(8,4:6)=-tr0(trd(1,3))*etp(5,:);else etg{5}(8,4:6)=-tr0(trd(2,3))*etp(6,:);end
    etg{5}(11,4:6)=-tr0(trd(2,4))*etp(8,:)-tr0(trd(1,4))*etp(7,:);
    if gpu;etg{5}=gpuArray(etg{5});end
end
