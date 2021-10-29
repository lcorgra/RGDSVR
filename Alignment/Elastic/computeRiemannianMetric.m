function E=computeRiemannianMetric(v0,L,sp,ups,de,F,FH,Lt)

%COMPUTERIEMANNIANMETRIC   Computes the Riemannian metric in the manifold
%of diffeomorphisms
%   E=RIEMANNIANMETRIC(V0,{L},{SP},{UPS},{DE},{F},{FH},{LT})
%   * V0 is the velocity field at t=0 in Fourier space
%   * L is a symmetric positive-definite differential operator
%   * {SP} is the spacing of the spatial grid
%   * {UPS} is a padding factor for circular convolutions
%   ** E is the computed metric
% 

gpu=isa(v0,'gpuArray');
N=size(v0);
ND=length(sp);

if nargin<4 || isempty(ups);ups=2;end
if nargin<5 || isempty(de);de=0;end
if nargin<8;Lt=[];end
if isscalar(ups);ups=ups*ones(1,ND);end

L=buildDifferentialOperator(L,N,sp,gpu);
N=N(1:ND);

if ~any(ups==0);M=N.*ups;
else M=N;
end
if nargin>=6;M=F;N=FH;end
if de==0
    if ~any(ups==0);w0=bsxfun(@times,v0,L);
    else w0=real(filtering(v0,L));
    end
    if ~isempty(Lt);w0=filtering(w0,Lt,1);end
    if ~any(ups==0);v0=mapSpace(v0,0,M,N);w0=mapSpace(w0,0,M,N);end    
    E=gather(w0(:)'*v0(:));
else
    E=v0(:);
    %v0=mapSpace(v0,0,M,N);
end
