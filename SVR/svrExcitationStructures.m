function svr=svrExcitationStructures(svr)

%SVREXCITATIONSTRUCTURES   Sets up the excitation structures for SVR: 
%Slice orders, number of packages, number of slices per package and 
%indexes of slices in package
%   SVR=SVREXCITATIONSTRUCTURES(SVR)
%   * SVR is a svr structure
%   ** SVR is a svr structure with excitation structures
%

gpu=useGPU;
svr.NV=length(svr.z);
svr.MPack=cell(svr.NV);
svr.y=svr.z;
[svr.MSY,svr.MSYorig]=deal(svr.MSZ);
[svr.MTY,svr.MTYorig]=deal(svr.MTZ);
svr.Par=svr.ParZ;

%RESAMPLING
for v=1:svr.NV
    if gpu;svr.y{v}=gpuArray(svr.y{v});end
    N=size(svr.y{v});N=N(1:2);
    M=round(N./(svr.Alg.MS./svr.MSY{v}(1:2)));
    svr.MSY{v}(1:2)=svr.MSY{v}(1:2).*N./M;
    [T,R,S]=factorizeHomogeneousMatrix(svr.MTY{v});
    for s=1:2;S(s,s)=svr.MSY{v}(s);end
    svr.MTY{v}=T*R*S;
    svr.y{v}=resampling(svr.y{v},M);
    svr.y{v}=gather(svr.y{v});
end 

%GEOMETRY
if ~isempty(svr.Alg.StackGeometry)
    if svr.Alg.StackGeometry~=1
        v=svr.Alg.StackGeometry;
        svr.y([v 1])=svr.y([1 v]);svr.MSY([v 1])=svr.MSY([1 v]);svr.MTY([v 1])=svr.MTY([1 v]);svr.MSYorig([v 1])=svr.MSYorig([1 v]);svr.MTYorig([v 1])=svr.MTYorig([1 v]);svr.Par([v 1])=svr.Par([1 v]);
    end
    [T,R]=factorizeHomogeneousMatrix(svr.MTY{1});
    for v=1:svr.NV;svr.MTY{v}=T*mldivide(R,mldivide(T,svr.MTY{v}));end 
    [svr.Geom.T,svr.Geom.R]=deal(T,R);
    svr.Geom.TR=svr.Geom.T*svr.Geom.R;
end

%EXCITATION
for v=1:svr.NV
    %SLICES
    svr.NSlices{v}=size(svr.y{v},3);
    if isempty(svr.Par{v}.SlOr);svr.slOrd{v}=1:svr.NSlices{v};else svr.slOrd{v}=svr.Par{v}.SlOr;end
    svr.MBFactor(v)=1;%Multiband factor    
    svr.MBBlock(v)=length(svr.slOrd{v});
    
    %PACKAGES    
    svr.P(v)=numel(find(diff(svr.slOrd{v})<0))+1;    
    if svr.P(v)==1
        svr.slPerPack{v}=svr.NSlices{v};
    else    
        svr.slPerPack{v}=find(diff(svr.slOrd{v})<0);    
        svr.slPerPack{v}=cat(1,svr.slPerPack{v}(1),diff(svr.slPerPack{v}),svr.MBBlock(v)-svr.slPerPack{v}(end));
        if svr.Alg.Debug;fprintf('Volume %d - Packages%s\n',v,sprintf(' %d',svr.slPerPack{v}));end    
    end

    svr.MPack{v}=real(zeros([1 1 svr.NSlices{v} 1 svr.P(v)],'like',svr.y{v}));%Mask of packages  
    if gpu;svr.MPack{v}=gpuArray(svr.MPack{v});end
    
    cont=1;
    for p=1:svr.P(v)
        svr.MSlices{v}{p}=real(zeros([1 1 svr.NSlices{v} 1 1 svr.slPerPack{v}(p)],'like',svr.MPack{v}));%3rd Dim slices-6th Dim excitations
        svr.slInPack{v}{p}=svr.slOrd{v}(cont:cont+svr.slPerPack{v}(p)-1);
        for m=1:svr.MBFactor(v);mb=(m-1)*svr.MBBlock(v);
            svr.MPack{v}(1,1,svr.slInPack{v}{p}+mb,1,p)=1;%3rd Dim slices-5th Dim packages
            for s=1:svr.slPerPack{v}(p)
                svr.MSlices{v}{p}(1,1,svr.slInPack{v}{p}(s)+mb,1,1,s)=1;       
            end                     
        end
        cont=cont+svr.slPerPack{v}(p);
    end
    
    %INTERLEAVES
    if isempty(find(diff(svr.slOrd{v})<0,1,'first'));svr=[];return;end%To detect issues and stop reconstruction
    svr.I(v)=svr.slOrd{v}(find(diff(svr.slOrd{v})<0,1,'first')+1)-svr.slOrd{v}(1); 
    if svr.I(v)<2
        error('Not contemplated interleaves:%s\n',sprintf(' %d',svr.slOrd{v}));
    else
        svr.slPerInte{v}=diff([1 find(ismember(svr.slOrd{v},2:svr.I(v)))' svr.MBBlock(v)+1]);
        if svr.Alg.Debug;fprintf('Volume %d - Interleaves%s\n',v,sprintf(' %d',svr.slPerInte{v}));end
    end
        
    svr.MInte{v}=real(zeros([1 1 svr.NSlices{v} 1 svr.I(v)],'like',svr.y{v}));%Mask of interleaves
    if gpu;svr.MInte{v}=gpuArray(svr.MInte{v});end
    
    cont=1;
    pior=1;
    for i=1:svr.I(v)
        svr.slInInte{v}{i}=svr.slOrd{v}(cont:cont+svr.slPerInte{v}(i)-1);
        pcont=0;
        np=0;
        pi=pior;
        while pcont<svr.slPerInte{v}(i)
            pcont=pcont+svr.slPerPack{v}(pi);
            np=np+1;
            pi=pi+1;
        end
        svr.paPerInte{v}(i)=np;
        svr.paInInte{v}{i}=pior:pi-1;
        pior=pi;
        for m=1:svr.MBFactor(v);mb=(m-1)*svr.MBBlock(v);
            svr.MInte{v}(1,1,svr.slInInte{v}{i}+mb,1,i)=1;%3rd Dim slices-5th Dim interleaves
        end
        cont=cont+svr.slPerInte{v}(i);
    end
    svr.UseInterleave=1;%To indicate that we estimate the interleaves to start with
end
