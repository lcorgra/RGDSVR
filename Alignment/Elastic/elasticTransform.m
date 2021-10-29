function x=elasticTransform(x,u,sp,ups,extrap)

%ELASTICTRANSFORM   Applies an elastic transform to x
%   [X,UR,URM,VR,VRM]=ELASTICTRANSFORM(X,U,SP,{UPS},{EXTRAP})
%   * U is the forward transform (any resolution)
%   * X is the image to be transformed
%   * SP is the spacing
%   * {UPS} is an upsampling factor to interpolate
%   * {EXTRAP} indicates whether to extrapolate, defaults to 1
%   ** X is the transformed image
% 

if nargin<4 || isempty(ups);ups=1;end
if nargin<5 || isempty(extrap);extrap=1;end

if ~iscell(u) && ~isstruct(u) && all(u(:)==0);return;end
ND=length(sp);
N=size(x);N=N(1:ND);
if isscalar(ups);ups=ups*ones(1,ND);end
gpu=isa(x,'gpuArray');
M=round(ups.*N);ups=M./N;
sp=sp./ups;      
rGrid=cell(1,ND);
for n=1:ND;rGrid{n}=single(0:M(n)-1)*sp(n);end
if extrap    
    lim=resPop(single(M.*sp),2,[],ND+1);
    Npad=ones(1,ND);
    for n=1:ND;rGrid{n}(M(n)+1)=lim(n);end
    if gpu;lim=gpuArray(lim);end
end
if gpu
    for n=1:ND;rGrid{n}=gpuArray(rGrid{n});end
end
r=gridv(rGrid);

if iscell(u)%We concatenate transforms (only 2 are allowed, either elastic-rigid or rigid-elastic or elastic-elastic
    for n=1:length(u)
        if ~isstruct(u{n});u{n}=resampling(u{n},M);end
    end       
    if isstruct(u{2}) && ~isstruct(u{1})%Elastic-rigid
        u=u{1}+precomputeDeformationSincRigidTransform(u{2}.rGrid(1:ND),u{2}.T,u{2}.di,u{2}.cGrid(1:ND),u{1});
    elseif isstruct(u{1}) && ~isstruct(u{2})%Rigid-elastic        
        u{1}=precomputeDeformationSincRigidTransform(u{1}.rGrid(1:ND),u{1}.T,u{1}.di,u{1}.cGrid(1:ND));        
        u=u{1}+elasticTransform(u{2},u{1},sp,1,extrap);        
    else
        u=u{1}+elasticTransform(u{2},u{1},sp,1,extrap);
    end
end
if isstruct(u);u=precomputeDeformationSincRigidTransform(u.rGrid,u.T,u.di,u.cGrid);end
u=resampling(u,M);
x=resampling(x,M);

if extrap
    x=padarray(x,Npad,'circular','post');
    u=padarray(u,Npad,'circular','post');
end    

ur=u+r;
if extrap;ur=bsxfun(@mod,ur,lim);end
NXS=size(x);
x=resSub(x,ND+1:max(numDims(x),ND+1));
for n=1:size(x,ND+1)
    if ND==2;x=dynInd(x,n,ND+1,interpn(dynInd(r,1,ND+1),dynInd(r,2,ND+1),dynInd(x,n,ND+1),dynInd(ur,1,ND+1),dynInd(ur,2,ND+1),'linear',0));
    else x=dynInd(x,n,ND+1,interpn(dynInd(r,1,ND+1),dynInd(r,2,ND+1),dynInd(r,3,ND+1),dynInd(x,n,ND+1),dynInd(ur,1,ND+1),dynInd(ur,2,ND+1),dynInd(ur,3,ND+1),'linear',0));
    end
end

x=reshape(x,NXS);
if extrap
    ROI=cell(1,ND);
    for n=1:ND;ROI{n}=1:M(n);end
    x=dynInd(x,ROI,1:ND);
end
x=resampling(x,N);
