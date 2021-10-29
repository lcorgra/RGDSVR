function [vt,vtg]=geodesicShooting(v0,sp,NT,L,G,ups)

%GEODESICSHOOTING   Computes the geodesic evolution of the velocity field V
%   VT=GEODESICSHOOTING(V0,SP,{NT},{L},{G},{UPS})
%   * V0 is the velocity field at t=0 in Fourier space
%   * SP is the spacing of the spatial grid
%   * {NT} is the number of points for integration from t=0 to t=1
%   * {L} is a symmetric positive-definite differential operator
%   * {G} is the gradient type, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions
%   * {UPS} is a padding factor for circular convolutions
%   ** VT is the evolution of the velocity field
%   ** VTG is the spatial gradient of the evolution of the velocity field
% 

gpu=isa(v0,'gpuArray');
N=size(v0);
ND=length(sp);

if nargin<3 || isempty(NT);NT=10;end
if nargin<4 || isempty(L);L=ones(N(1:ND),'like',v0);end
if nargin<5 || isempty(G);G='1stFiniteDiscreteCentered';end
if nargin<6 || isempty(ups);ups=2;end
if isscalar(ups);ups=ups*ones(1,ND);end

L=buildDifferentialOperator(L,N,sp,gpu);
if ~any(ups==0)
    M=round(N(1:ND).*ups);ups=M./N(1:ND);
    F=buildMapSpace(v0,1,M,N);FH=buildMapSpace(v0,0,M,N);
    G=buildGradientOperator(G,N,sp,gpu);
else
    M=N(1:ND);
    F=buildMapSpace(v0,1,M,N);FH=buildMapSpace(v0,0,M,N);
    G=[];
    v0=mapSpace(v0,1,F,FH);
end
    
vt=zeros([N 1 NT],'like',v0);
vt=dynInd(vt,1,ND+3,v0);
if nargout>=2;vtg=zeros([N ND NT],'like',v0);end
NTV=2:NT;
dt=1/(length(NTV)+1);%dt=1/length(NTV);

for n=NTV%We assume velocities are at 1/(2*NT):1/NT:1-1/(2*NT)
    [vd,vg]=adAdjointOperator(v0,v0,sp,L,G,ups,F,FH);
    if nargout>=2;vtg=dynInd(vtg,n-1,ND+3,vg);end    
    v0=v0-vd*dt;
    vt=dynInd(vt,n,ND+3,v0);
end
if nargout>=2
    [~,vg]=adAdjointOperator(v0,v0,sp,L,G,ups,F,FH);
    vtg=dynInd(vtg,NT,ND+3,vg);
end

if any(ups==0);vt=mapSpace(vt,0,F,FH);end
