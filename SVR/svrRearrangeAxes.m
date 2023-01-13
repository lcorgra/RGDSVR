function svr=svrRearrangeAxes(svr)

%SVRREARRANGEAXES   Sets up the axes for SVR
%Rearrange the images so that the arrays are as close as possible to the
%reference image, store new slice and in plane dimensions
%   SVR=SVRREARRANGEAXES(SVR)
%   * SVR is a svr structure
%   ** SVR is a svr structure with rearranged axes
%

svr.NV=length(svr.y);
svr.NY=cell(1,svr.NV);
svr.id=zeros(svr.NV,3);%Dimensions after permuting the axes
for v=1:svr.NV   
    [~,R]=factorizeHomogeneousMatrix(svr.MTY{v});
    RU=generatePrincipalAxesRotations(1);%Recently corrected bug
    d=rotationDistance(R(1:3,1:3),RU);
    [~,id]=min(d);    
    MTn=RU(:,:,id);
    
    [~,svr.id(v,:)]=max(abs(MTn),[],1);%Closest material dimension for each spatial dimension    
    ord=sign(indDim(MTn,svr.id(v,:),1));%Direction of each spatial dimension
    for n=1:3%We flip the dimensions
        if ord(n)<0
            svr.y{v}=flip(svr.y{v},n);
            svr.MPack{v}=flip(svr.MPack{v},n);
            svr.MInte{v}=flip(svr.MInte{v},n);
            for p=1:svr.P(v);svr.MSlices{v}{p}=flip(svr.MSlices{v}{p},n);end
            Nf=zeros(4,1);Nf(n)=size(svr.y{v},n)-1;Nf(4)=1;
            svr.MTY{v}(:,4)=svr.MTY{v}*Nf;
            svr.MTY{v}(:,n)=-svr.MTY{v}(:,n);
        end
    end%We permute the dimensions
    svr.y{v}=ipermute(svr.y{v},svr.id(v,:));
    svr.MPack{v}=ipermute(svr.MPack{v},[svr.id(v,:) 4 5]);
    svr.MInte{v}=ipermute(svr.MInte{v},[svr.id(v,:) 4 5]);
    for p=1:svr.P(v);svr.MSlices{v}{p}=ipermute(svr.MSlices{v}{p},[svr.id(v,:) 4 5 6]);end
    svr.MSY{v}(svr.id(v,:))=svr.MSY{v};
    svr.MTY{v}(1:3,svr.id(v,:))=svr.MTY{v}(1:3,1:3);
    svr.NY{v}=size(svr.y{v})';%Sizes
end
svr.MSY=cat(3,svr.MSY{:});svr.MSY=permute(svr.MSY,[2 1 3]);
svr.MTY=cat(3,svr.MTY{:});
svr.NY=cat(3,svr.NY{:});
