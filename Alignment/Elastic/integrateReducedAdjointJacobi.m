function dv=integrateReducedAdjointJacobi(dv,vt,sp,L,G,ups)

%INTEGRATEREDUCEDADJOINTJACOBI   Integrates the reduced adjoint Jacobi
%equations to transport gradient fields from t=1 to t=0
%   DV=INTEGRATEREDUCEDADJOINTJACOBI(DV,VT,SP,{L},{G},{UPS})
%   * DV is the gradient at t=1
%   * VT are the velocity fields at different instants in time
%   * SP is the spacing of the spatial grid
%   * {L} is a symmetric positive-definite differential operator
%   * {G} is the gradient type, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions
%   * {UPS} is a padding factor for circular convolutions
%   ** DV is the gradient at t=0
% 

gpu=isa(dv,'gpuArray');
ND=length(sp);
N=size(vt);N(end+1:ND+3)=1;

if nargin<4 || isempty(L);L=ones(N(1:ND),'like',dv);end
if nargin<5 || isempty(G);G='1stFiniteDiscreteCentered';end
if nargin<6 || isempty(ups);ups=2;end
if isscalar(ups);ups=ups*ones(1,ND);end

L=buildDifferentialOperator(L,N,sp,gpu);

if ~any(ups==0)
    M=round(N(1:ND).*ups);ups=M./N(1:ND);
    F=buildMapSpace(dv,1,M,N);FH=buildMapSpace(dv,0,M,N);
    G=buildGradientOperator(G,N,sp,gpu);
else
    M=N(1:ND);
    F=buildMapSpace(dv,1,M,N);FH=buildMapSpace(dv,0,M,N);
    G=[];    
    dv=mapSpace(dv,1,F,FH);vt=mapSpace(vt,1,F,FH);
end

NT=N(ND+3);
NTV=NT:-1:1;
dt=1/length(NTV);
U=dv;dv(:)=0;
for n=NTV
    v=dynInd(vt,n,ND+3);
    if n==NTV(1);dv=U;else dv=dv+(U+adAdjointOperator(dv,v,sp,L,G,ups,F,FH)-adAdjointOperator(v,dv,sp,L,G,ups,F,FH))*dt;end
    if n~=NTV(end);U=U+adAdjointOperator(v,U,sp,L,G,ups,F,FH)*dt;end
end
if any(ups==0);dv=mapSpace(dv,0,F,FH);end
