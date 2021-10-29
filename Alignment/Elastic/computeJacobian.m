function [Jd,J]=computeJacobian(u,sp,N,ups,extrap,G)

%COMPUTEJACOBIAN   Computes the Jacobian of a deformation field
%   [JD,J]=COMPUTEJACOBIAN(U,SP,{N},{UPS},{EXTRAP})
%   * U is the forward transform (any resolution)
%   * SP is the spacing
%   * {N} is the resolution at which to compute the Jacobian
%   * {UPS} is an upsampling factor to interpolate
%   * {EXTRAP} indicates whether to extrapolate
%   * {G} is the gradient type for the Jacobian, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions
%   ** JD is the determinant of the Jacobian
%   ** J is the full Jacobian
% 

if nargin<3 || isempty(N);N=size(u);end
if nargin<4 || isempty(ups);ups=1;end
if nargin<5 || isempty(extrap);extrap=1;end
if nargin<6 || isempty(G);G='1stFiniteDiscreteCentered';end

gpu=isa(u,'gpuArray');
ND=length(sp);
if isscalar(ups);ups=ups*ones(1,ND);end
u=resampling(u,N(1:ND));
N=size(u);N=N(1:ND);

typ=1;%Only upsampling the determinant computation
if ~any(ups==0)
    if typ==1
        M=round(ups.*N);   
    else
        M=round(ups.*N);ups=M./N;
        sp=sp./ups;
    end
else
    M=N;
end
for n=1:ND;rGrid{n}=single(1:N(n))*sp(n);end        

Npad=zeros(1,ND);%Padding for periodic boundary conditions
if extrap
    ua=abs(u);
    for n=1:ND
        Npad(n)=2*(ceil(gather(multDimMax(dynInd(ua,{[1 N(n)],n},[n ND+1]),1:ND)))+1);
        Npad(n)=min(Npad(n),N(n));
        rG=(1:Npad(n))*sp(n);
        rGrid{n}=cat(2,-flip(rG)+rGrid{n}(1),rGrid{n},rGrid{n}(N(n))+rG);
    end
    u=padarray(u,Npad,'circular','both');
    %fprintf('Padding:%s\n',sprintf(' %d',Npad));
end
if gpu
    for n=1:ND;rGrid{n}=gpuArray(rGrid{n});end
end    
r=gridv(rGrid);
if ~any(ups==0);J=deformationGradientTensor(u+r,sp,G);
else J=deformationGradientTensorSpace(u+r,sp);
end
if extrap
    NN=size(J);
    ROI=cell(1,ND);
    for n=1:ND;ROI{n}=1+Npad(n):NN(n)-Npad(n);end
    J=dynInd(J,ROI,1:ND);
end

if typ==1;Jd=resampling(J,M);
else Jd=J;
end

if ND==2;Jd=ipermute(det2x2m(permute(Jd,[3 4 1 2])),[3 4 1 2]);
else Jd=ipermute(det3x3m(permute(Jd,[4 5 1 2 3])),[4 5 1 2 3]);
end
Jd=resampling(Jd,N);
if ~typ
    if nargout>=2;J=resampling(J,N);end
end
