function svr=svrSetUp(svr)

%SVRSETUP   Sets up the SVR information
%   SVR=SVRSETUP(SVR)
%   * SVR is a svr structure
%   ** SVR is a svr structure with structures required for reconstruction
%

gpu=useGPU;

%SLICE ORDERS / NUMBER OF PACKAGES / NUMBER OF SLICES PER PACKAGE / INDEXES OF SLICES IN PACKAGE
svr=svrExcitationStructures(svr);
if isempty(svr);return;end

%ARRANGEMENT OF AXES
svr=svrRearrangeAxes(svr);

%1) GENERIC CENTER OF FOV
svr.cFOV=(svr.NY)/2;svr.cFOV(4,:,:)=1;
svr.cFOV=matfun(@mtimes,svr.MTY,svr.cFOV);%Center of the FOV in spatial coordinates
svr.cFOV=median(svr.cFOV,3);%Median of center of different stacks

%2) GENERIC STACKS FOV
svr.FOV=zeros([3 2 svr.NV]);%Axis - Lower/Upper limit - Stack
svr.FOV(:,2,:)=svr.NY-1;%Upper limit
svr.FOV=indDim(svr.FOV,ind2subV(2*ones(1,3),1:2^3)',2);svr.FOV(4,:,:)=1;%FOV corners of different stacks
svr.FOV=matfun(@mtimes,svr.MTY,svr.FOV);%Dimensions 4-8-NV-FOV corners in spatial coordinates
svr.FOVLim=cat(2,min(svr.FOV,[],2),max(svr.FOV,[],2));svr.FOVLim=svr.FOVLim(1:3,:,:);%FOV limits of different stacks
svr.FullFOVLim=cat(2,min(svr.FOVLim(:,1,:),[],3),max(svr.FOVLim(:,2,:),[],3));%Minimum FOV that contains all stacks
svr.FullFOVExt=svr.FullFOVLim(:,2)-svr.FullFOVLim(:,1);%Extent of the FOV

if ~isempty(svr.Alg.MS);svr.MSX=svr.Alg.MS;else svr.MSX=svr.Alg.Resol;end%Either as parameter or as the absolute minimum of all resolutions
svr.MSX=svr.MSX*ones(3,1);%Spacing of reconstructions
svr.FullFOVExt=ceil(svr.FullFOVExt./svr.MSX);%Full FOV till here

svr.FullFOVExt=svr.FullFOVExt+1-mod(svr.FullFOVExt,2);
svr.FullFOVExt=svr.MSX.*svr.FullFOVExt;
[svr.FullFOVExt,svr.FullFOVLim]=mapOddFOV(svr.FullFOVExt,svr.MSX,svr.cFOV);%Force odd grid sizes
[svr.NX,svr.NXFull,svr.MTX,svr.MSX]=dimensionsFOV(svr.FullFOVLim,svr.MSX,svr.Alg.Resol);%Compute grid sizes
svr.MTXWrite=svr.Geom.TR*mldivide(svr.Geom.T,svr.MTX);

%3) FOV CROPPING
if isfield(svr,'R') && ~isempty(svr.R)   
    xaux=zeros(svr.NX','like',svr.y{1});
    svr.RX=mapVolume(svr.R,xaux,svr.MTR,svr.MTXWrite,[],[],0,'nearest');
    ROI=double(computeROI(svr.RX,ceil(svr.Alg.SecurityFOV./svr.MSX)));%20 mm security margin
    svr.cFOV(1:3,1,1)=(mean(ROI(:,1:2)-0.5*(1-mod(diff(ROI(:,1:2),1,2),2)),2)-1);%Center of FOV for a given ROI
    svr.cFOV=svr.MTX*svr.cFOV;%Center of FOV in spatial coordinates
    svr.FullFOVExt=ceil((diff(ROI(:,1:2),1,2)+1));%Reduced FOV till here
    
    svr.FullFOVExt=svr.FullFOVExt+1-mod(svr.FullFOVExt,2);
    svr.FullFOVExt=svr.MSX.*svr.FullFOVExt;   
    [svr.FullFOVExt,svr.FullFOVLim]=mapOddFOV(svr.FullFOVExt,svr.MSX,svr.cFOV);
    [svr.NX,svr.NXFull,svr.MTX,svr.MSX]=dimensionsFOV(svr.FullFOVLim,svr.MSX,svr.Alg.Resol);    
    svr.MTXWrite=svr.Geom.TR*mldivide(svr.Geom.T,svr.MTX);
    xaux=zeros(svr.NX','like',svr.y{1});
    svr.RX=mapVolume(svr.R,xaux,svr.MTR,svr.MTXWrite,[],[],0,'nearest');
end

%4) TRANSFORMS FROM SPATIAL TO MATERIAL
perma=[2 3 4 5 6 1];
%A) Rotation matrix, angles of rotation and rotation grid
svr.AcqToRec=matfun(@mldivide,svr.MTX,svr.MTY);
[~,svr.AcqToRecRot,~]=factorizeHomogeneousMatrix(svr.AcqToRec);
svr.AcqToRecRot=svr.AcqToRecRot(1:3,1:3,:);
%B) Coordinates of acquired data in reconstructed resolution
svr.cFOVRec=matfun(@mldivide,svr.MTX,svr.cFOV);
svr.cFOVRec=svr.cFOVRec(1:3)+1;%+1;%Center of coordinates in pixels, reconstruction space
svr.cFOVAcq=matfun(@mldivide,svr.MTY,svr.cFOV);%Center of coordinates in pixel coordinates, acquisition space
svr.cFOVAcq=svr.cFOVAcq(1:3,:,:);%Center of coordinates in pixels, acquisition space
svr.cFOVAcq=bsxfun(@times,svr.cFOVAcq,bsxfun(@rdivide,svr.MSY,svr.MSX));%Center of coordinates in pixels at the reconstructed resolution
svr.cFOVAcq=svr.cFOVAcq+1;%+1;
svr.IndAcq=svr.cFOVAcq-svr.cFOVRec;%Starting index on the acquisition space (referred to 0)
svr.IndAcqRound=round(svr.IndAcq);
if isfield(svr,'IndAcqShift');svr.IndAcqShift0=svr.IndAcqShift;end
svr.IndAcqShift=svr.IndAcq-svr.IndAcqRound;%This shift should be applied as a translation after rotation, but we haven't coded it, we let the method learn it
svr.IndAcqShift=bsxfun(@times,svr.IndAcqShift,svr.MSX);
svr.IndAcqShift=matfun(@mldivide,svr.AcqToRecRot,svr.IndAcqShift);
%C) Rearranging
svr.NXY=round(bsxfun(@rdivide,bsxfun(@times,svr.NY,svr.MSY),svr.MSX));%Size of acquired data when resampled to reconstructed resolution
svr.NX=svr.NX';svr.MSX=svr.MSX';svr.NY=permute(svr.NY,[3 1 2]);svr.MSY=permute(svr.MSY,[2 1 3]);svr.NXY=permute(svr.NXY,[3 1 2]);svr.cFOVRec=svr.cFOVRec';

%5) RECONSTRUCTED DATA AND STRUCTURES FOR RECONSTRUCTION
if isfield(svr,'x');svr.x=resampling(svr.x,svr.NX,0,ones(1,3));else svr.x=zeros(svr.NX,'like',svr.y{1});end
if gpu;svr.x=gpuArray(svr.x);end
svr.M=svr.x;svr.M(:)=1;
[svr.xx,svr.yy,svr.E]=deal([]);%For iterating the reconstruction

%6) STRUCTURES FOR DEFORMABLE REGISTRATION
FOV=svr.MSX.*svr.NX;
svr.Alg.NT=ceil(FOV./svr.Alg.NTRes);%NTResmm deformation resolution
svr.Alg.NT=svr.Alg.NT+1-mod(svr.Alg.NT,2);%Odd
if svr.Alg.Debug;fprintf('Motion size:%s\n',sprintf(' %d',svr.Alg.NT));end
svr.MST=FOV./svr.Alg.NT;
svr.PreTr=isfield(svr,'TV');%Whether a transformation for stacks already exists
svr.MotEst=0;%Level at which to start the alternate minimization
%A) Volume transforms
if ~svr.PreTr;svr.TV=zeros([ones(1,4) svr.NV 6],'single');end
if ~svr.PreTr || (isfield(svr,'DV') && any(svr.Alg.NT~=multDimSize(svr.DV,1:3)))
    svr.DV=zeros([svr.Alg.NT 3 svr.NV],'single');%Non-rigid transform
    if gpu;svr.DV=gpuArray(svr.DV);end
    svr.DVJac=dynInd(svr.DV,1,4);svr.DVJac(:)=1;
    svr.DVH=svr.DV;
    svr.DVRes=svr.DVJac;svr.DVRes(:)=0;
    if ~svr.PreTr;svr.VV=svr.DV;else svr.VV=resampling(svr.VV,svr.Alg.NT,1);end%Non-rigid velocity        
    if svr.Alg.Ups(1)~=0;svr.VV=complex(svr.VV);end
end
svr.Ld=buildDifferentialOperator(svr.Alg.RegDiff,svr.Alg.NT,svr.MST,gpu);
svr.GdT=buildGradientOperator('1stFiniteDiscreteCentered',svr.Alg.NT,svr.MST,gpu);
svr.GdX=buildGradientOperator('1stFiniteDiscreteCentered',svr.NX,svr.MSX,gpu);
if svr.Alg.Debug;fprintf('Maximum attenuation of Hilbert gradient: %.2f\n',max(svr.Ld(:)));end
for v=1:svr.NV
    %B) Crop to valid indexes
    for s=1:3
        svr.vAcq{v}{s}=svr.IndAcqRound(s,1,v)+1:floor(svr.IndAcq(s,1,v))+svr.NX(s);
        svr.vRec{v}{s}=find(svr.vAcq{v}{s}>=1 & svr.vAcq{v}{s}<=svr.NXY(v,s));
        svr.vAcq{v}{s}=svr.vAcq{v}{s}(svr.vRec{v}{s});        
    end        
    %C) Rigid transforms
    if ~svr.PreTr
        svr.TI{v}=zeros([ones(1,4) svr.I(v) 6]);%Interleave transforms
        svr.TP{v}=zeros([ones(1,4) svr.P(v) 6]);%Package transforms
        for p=1:svr.P(v);svr.TE{v}{p}=zeros([ones(1,4) svr.slPerPack{v}(p) 6]);end%Slice transforms
        svr.TV(1,1,1,1,v,4:6)=-permute(rotm2eul(svr.AcqToRecRot(:,:,v))',perma);%Convert axes rotation matrix to Euler angles  
        svr.TV(1,1,1,1,v,1:3)=permute(svr.IndAcqShift(:,1,v),perma);
    else
        svr.TV(1,1,1,1,v,1:3)=svr.TV(1,1,1,1,v,1:3)+permute(svr.IndAcqShift(:,1,v)-svr.IndAcqShift0(:,1,v),perma);
    end    
end
%D) Deformable transforms
if ~svr.PreTr || any(svr.Alg.NT~=multDimSize(svr.DP{1},1:3))
    for v=1:svr.NV
        %Interleaves
        svr.DI{v}=zeros([svr.Alg.NT 3 svr.I(v)],'single');%Package transforms
        if gpu;svr.DI{v}=gpuArray(svr.DI{v});end
        svr.DIJac{v}=dynInd(svr.DI{v},1,4);svr.DIJac{v}(:)=1;
        svr.DIH{v}=svr.DI{v};        
        svr.DIRes{v}=svr.DIJac{v};svr.DIRes{v}(:)=0;
        if ~svr.PreTr;svr.VI{v}=svr.DI{v};else svr.VI{v}=resampling(svr.VI{v},svr.Alg.NT,1);end%Non-rigid velocity               
        if svr.Alg.Ups(1)~=0;svr.VI{v}=complex(svr.VI{v});end        
        %Packages
        svr.DP{v}=zeros([svr.Alg.NT 3 svr.P(v)],'single');%Package transforms
        if gpu;svr.DP{v}=gpuArray(svr.DP{v});end
        svr.DPJac{v}=dynInd(svr.DP{v},1,4);svr.DPJac{v}(:)=1;
        svr.DPH{v}=svr.DP{v};        
        svr.DPRes{v}=svr.DPJac{v};svr.DPRes{v}(:)=0;
        if ~svr.PreTr;svr.VP{v}=svr.DP{v};else svr.VP{v}=resampling(svr.VP{v},svr.Alg.NT,1);end%Non-rigid velocity               
        if svr.Alg.Ups(1)~=0;svr.VP{v}=complex(svr.VP{v});end                        
    end
elseif ~isfield(svr,'DI')
    svr.UseInterleave=0;
end
if isfield(svr,'DV');[svr.DV,svr.DVH,svr.DVJac,svr.DVRes,svr.VV]=computeDeformableTransforms(svr.VV,svr.Alg.NG,svr.Ld,svr.GdT,svr.MST,svr.Alg.Ups([1 4]));end
if isfield(svr,'DI');[svr.DI,svr.DIH,svr.DIJac,svr.DIRes,svr.VI]=computeDeformableTransforms(svr.VI,svr.Alg.NG,svr.Ld,svr.GdT,svr.MST,svr.Alg.Ups([1 4]));end
[svr.DP,svr.DPH,svr.DPJac,svr.DPRes,svr.VP]=computeDeformableTransforms(svr.VP,svr.Alg.NG,svr.Ld,svr.GdT,svr.MST,svr.Alg.Ups([1 4]));

%7) SLICE WEIGHTS/DISCARDING
if ~isfield(svr,'W')
    svr.W=cell(1,svr.NV);
    for v=1:svr.NV
        svr.W{v}=real(ones(svr.NY(v,:),'like',svr.y{v}));
        if gpu;svr.W{v}=gpuArray(svr.W{v});end
    end
end
if ~isfield(svr,'D')
    svr.D=cell(1,svr.NV);
    for v=1:svr.NV
        NW=ones(1,3);NW(svr.id(v,3))=svr.NY(v,svr.id(v,3));
        svr.D{v}=real(zeros(NW,'like',svr.W{v}));
        if gpu;svr.D{v}=gpuArray(svr.D{v});end
    end
end

%8) SLICE PROFILES
svr.SlPr=cell(1,svr.NV);
for v=1:svr.NV         
    NSlPr=svr.NXY(v,svr.id(v,3));
    sigma=svr.Alg.FWHM/(2*sqrt(2*log(2)));%For FWHM as a ratio of slice thickness and slice separation    
    sigma=sigma*svr.NXY(v,svr.id(v,3))/svr.NY(v,svr.id(v,3));
    NSl=ones(1,3);NSl(svr.id(v,3))=2*NSlPr;
    kGrid=generateGrid(NSl,gpu,pi,ceil((NSl+1)/2));   
    svr.SlPr{v}=exp(-(kGrid{svr.id(v,3)}.^2)*(2*sigma^2));    
    svr.SlPr{v}=ifftshift(svr.SlPr{v},svr.id(v,3));         
    svr.SlPr{v}=sqrt(2*NSlPr)*svr.SlPr{v}/norm(svr.SlPr{v}(:));
    svr.SlPr{v}=svr.SlPr{v}/svr.SlPr{v}(1);%Normalize for same average
    svr.SlPr{v}=dynInd(svr.SlPr{v},1:NSlPr,svr.id(v,3));                  
end
for v=1:svr.NV
    NSlPr=svr.NXY(v,svr.id(v,3));
    svr.SlInterleaveFilter{v}=resPop(fftGPU(buildFilter(NSlPr,'tukeyIso',2*svr.I(v)/NSlPr,gpu,1),1),1,[],svr.id(v,3));
end

%9) WINDOWING
[~,svr.NYUniqueInd,svr.NYFullInd]=unique(svr.NY,'rows');
NVU=length(svr.NYUniqueInd);
svr.H=cell(2,NVU);
for v=1:NVU;svr.H{1}{v}=fftshift(buildFilter(svr.NY(svr.NYUniqueInd(v),:),'tukeyC',ones(1,3),0,svr.Alg.Windowing(1)*ones(1,3)));end
for n=2:3
    if svr.Alg.Windowing(n)>1;factWindowing(n,:)=2*(svr.Alg.Windowing(n)./svr.MSX)./svr.NX;else factWindowing(n,:)=svr.Alg.Windowing(n)*ones(1,3);end
end
svr.Hx{2}=fftshift(buildFilter(svr.NX,'tukeyC',(1-factWindowing(3,:)),gpu,factWindowing(2,:)./(1-factWindowing(3,:))));%For motion correction
svr.Hx{1}=fftshift(buildFilter(svr.NX,'tukeyC',ones(1,1),gpu,factWindowing(3,:)));%For reconstruction

%10) TRANSFORM GRIDS
[svr.rGrid,svr.kGrid,svr.rkGrid,~,svr.cGrid]=generateTransformGrids(svr.NX.*svr.MSX,gpu,svr.NX,svr.cFOVRec,1);
svr.FT=buildStandardDFTM(svr.NX,1,gpu);

%11) REGULARIZER
%A) Standard
if svr.Alg.RegularizationType<2 || ~(svr.Alg.MS/svr.Alg.Resol<1.5 && svr.Alg.MS/svr.Alg.Resol>0.75)%STANDARD    
    NG=length(svr.Alg.RegFracOrd);
    indNz=find(svr.Alg.RegFracOrd~=0,1,'first');
    for g=1:NG
        svr.F{g}=buildFilter(2*svr.NX,'FractionalFiniteDiscreteIsoNorm',ones(1,3),gpu,svr.Alg.RegFracOrd(g),1);
        if svr.Alg.RegFracOrd(g)~=0;svr.F{g}(1)=0;end
        svr.F{g}=svr.Alg.Ti(g)*abs(svr.F{g}).^2;
        if g>indNz;svr.F{indNz}=svr.F{indNz}+svr.F{g};end
    end
    if isempty(indNz);indNz=1;end
    svr.F=svr.F(1:indNz);
    svr.xz=[];
%B) Deep Decoder
else
    svr.F=[];
    svr.dd.parNet.N=svr.NX;%Size of the image
    svr.dd.parNet.ND=3;%Dimensions of the image
    svr.dd.Rtarget=2;%Compression ratio
    svr.dd.M=1;%Number of channels of image
    [svr.dd.parNet,svr.dd.parFit,svr.dd.parDat,svr.dd.Rtarget]=parametersDeepDecoder(svr.dd.parNet,svr.dd.Rtarget,[],svr.Alg.Debug);
    svr.dd.parNet.L=svr.dd.parNet.fact_flatten*max(5+ceil(log2(prod(svr.dd.parNet.N)^(1/svr.dd.parNet.ND)))-9,3);              
    if ismember(svr.dd.parNet.wav_over,0);svr.dd.parNet.L=svr.dd.parNet.L-svr.dd.parNet.wav_L;end
    svr.dd.parNet.NRe=circshift([(2^(1/svr.dd.parNet.fact_flatten))*ones(1,svr.dd.parNet.L) 1 1],1-svr.dd.parNet.upsample_first);            
    svr.dd.parNet.NCo=svr.dd.parNet.NCo*ones(1,svr.dd.parNet.L+2);
    [svr.dd.parNet.K,svr.dd.R]=channelsDeepDecoder(svr.dd.parNet.L,svr.dd.parNet.N,svr.dd.M,svr.dd.Rtarget,svr.dd.parNet);
    if svr.Alg.Debug
        fprintf('DD number of levels: %d\n',svr.dd.parNet.L);
        fprintf('DD number of channels:%s\n',sprintf(' %d',svr.dd.parNet.K));
        fprintf('DD Effective ratio of compression: %.2f\n',svr.dd.R); 
    end  
    svr.dd.net=svr.df.arch.DeepDecoderNetwork(svr.dd.parNet);  
    svr.dd.parFit.epochs=svr.Alg.Epochs;
    svr.dd.parFit.verbose_frequency=1e8;
    svr.dd.random_input_sigma=0;
    svr.dd.parFit.file_name_ou='';    
    NEff=ceil(svr.dd.parNet.N/prod(svr.dd.parNet.NRe));KEff=svr.dd.parNet.K(1);
    svr.dd.x=rand([NEff KEff],'single')-(1-svr.dd.parDat.typ_input(1))*0.5;
    if svr.dd.parDat.typ_input(2)==2;svr.dd.x=svr.dd.x/10;end
    if svr.dd.parDat.typ_input(2)==1;svr.dd.x=svr.dd.x/sqrt(12);end
    svr.dd.x=gather(permute(svr.dd.x,[svr.dd.parNet.ND+2 svr.dd.parNet.ND+1 1:svr.dd.parNet.ND]));%Samples / Channels / Space                               
    svr.xz=svr.x;svr.xz(:)=0;
end

%12) FOR FRACTIONAL FINITE DIFFERENCE MOTION ESTIMATION
svr.G=cell(1,2);
for s=1:2
    svr.G{s}=cell(1,svr.NV);
    if svr.Alg.MotFracOrd(s)~=0
        for v=1:svr.NV
            NG=svr.NY(v,:);NG(svr.id(v,3))=1;
            if svr.Alg.MotJointFracOrd%Joint
                svr.G{s}{v}=buildFilter(NG,'FractionalIso',ones(1,3),gpu,svr.Alg.MotFracOrd(s));
            else%Separable (quicker)
                c=1;
                for m=1:3
                    if m~=svr.id(v,3)
                        NGF=ones(1,3);NGF(m)=NG(m);
                        svr.G{s}{v}=buildFilter(NG,'FractionalIso',ones(1,3),gpu,svr.Alg.MotFracOrd(s));
                        c=c+1;
                    end
                end
            end
        end
    end
end

function [FullFOVExt,FullFOVLim]=mapOddFOV(FullFOVExt,MSX,cFOV)
    FullFOVExt=FullFOVExt./MSX;
    FullFOVExt=FullFOVExt+1-mod(FullFOVExt,2);
    FullFOVExt=MSX.*FullFOVExt;
    FullFOVLim=bsxfun(@plus,cFOV(1:3,1),cat(2,-FullFOVExt/2,FullFOVExt/2));
    fprintf('Reconstructed FOV:%s\n',sprintf(' %.2f',FullFOVExt));
end

function [NX,NXFull,MTX,MSX]=dimensionsFOV(FullFOVLim,MSX,Resol)
    NX=round((FullFOVLim(:,2)-FullFOVLim(:,1))./MSX);%Size of the reconstruction FOV
    NXFull=round((FullFOVLim(:,2)-FullFOVLim(:,1))./(Resol*ones(1,3)));%Size of the reconstruction FOV
    MTX=eye(4);
    MTX(1:3,4)=FullFOVLim(:,1);
    MSX=(FullFOVLim(:,2)-FullFOVLim(:,1))./NX;
    MTX(1:3,1:3)=MSX(1)*eye(3);
    fprintf('Reconstructed grid size:%s\n',sprintf(' %d',NX));
end

end