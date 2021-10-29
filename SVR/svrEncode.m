function svr=svrEncode(svr,computeRes,vU,useT)

%SVRENCODE   Encoding operator for SVR
%   SVR=SVRENCODE(SVR,{COMPUTERES},{VU},{USET})
%   * SVR is a svr structure
%   * {COMPUTERES} serves to compute residuals, it defaults to 0, if 2 it
%   also computes the discard factors, -1 it inverts residuals
%   * {VU} serves to encode only a specific view/package/slice
%   * {USET} serves to indicate that the encoding includes the transform
%   ** SVR is a svr structure after encoding
%

if nargin<2;computeRes=[];end
if nargin<3 || isempty(vU);vU=[];end
if nargin<4 || isempty(useT);useT=5;end

mirr=2*ones(1,3);

if svr.UseInterleave==1
    TPI=svr.TI;
    DPI=svr.DI;
    MPI=svr.MInte;
    slPerPI=svr.slPerInte;
    NPI=svr.I;
else
    TPI=svr.TP;
    DPI=svr.DP;
    MPI=svr.MPack;
    slPerPI=svr.slPerPack;
    NPI=svr.P;
end

for v=1:svr.NV 
    dsl=svr.id(v,3);
    if length(vU)<1 || v==vU(1)
        x=svr.xx;
        
        %PER-VOLUME TRANSFORM
        if useT>4;x=real(sincRigidTransformQuick(x,precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(svr.TV,v,5),1,0,[],1,svr.cGrid),1,svr.FT));end        
        if useT>3 && isfield(svr,'DV');x=elasticTransform(x,dynInd(svr.DV,v,5),svr.MSX,svr.Alg.Ups(3));end

        
        xc=zeros(svr.NY(v,:),'like',x);        
        useP=(any(TPI{v}(:)~=0) && useT>2) || (any(DPI{v}(:)~=0) && useT>1);
        for p=1:NPI(v)
            if length(vU)<2 || p==vU(2)
                NS=slPerPI{v}(p);
                useS=any(svr.TE{v}{p}(:)~=0) && useT>0;
                if ~useS;NS=1;end
                if useS;useP=1;end
                
                xp=x;
                %PER-PACK TRANSFORM                
                if useT>2;xp=elasticTransform(xp,dynInd(DPI{v},p,5),svr.MSX,svr.Alg.Ups(3));end
                if useT>1;xp=real(sincRigidTransformQuick(xp,precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(TPI{v},p,5),1,0,[],1,svr.cGrid),1,svr.FT));end                

                xb=zeros(svr.NY(v,:),'like',xp);
                for s=1:NS
                    if length(vU)<3 || s==vU(3)  
                        xs=xp;
                        %PER-SLICE TRANSFORM            
                        %if useS;xs=real(sincRigidTransformQuick(xs,precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(svr.TE{v}{p},s,6),1,0,[],1,svr.cGrid),1,svr.FT));end                        

                        %EXTRACT ARRAYS
                        NNXX=size(xs);NNXX(end+1:4)=1;
                        xa=zeros([svr.NXY(v,:) NNXX(4:end)],'like',xs);        
                        xa=dynInd(xa,svr.vAcq{v},1:3,dynInd(xs,svr.vRec{v},1:3));

                        %SLICE PROFILE
                        xa=filtering(xa,svr.SlPr{v},1);
                        
                        %SLICE DELTA-SAMPLING
                        xa=fold(xa,dsl,svr.NXY(v,dsl),svr.NY(v,dsl),[],[],1);

                        %RESAMPLING
                        xa=real(resampling(xa,svr.NY(v,:),[],mirr));

                        %PER-SLICE MASKING
                        if useS;xb=xb+bsxfun(@times,xa,dynInd(svr.MSlices{v}{p},s,6));else xb=xa;end
                    end
                end;xa=[];xs=[];

                %PER-PACK MASKING
                if useP;xc=xc+bsxfun(@times,xb,dynInd(MPI{v},p,5));else xc=xb;end
                if ~useP;break;end
            end
        end;xb=[];xp=[];

        %COMPUTE RESIDUALS
        if ismember(2,computeRes)
            svr.D{v}=multDimSum(abs(xc).^2,setdiff(1:3,dsl));
            meaD=mean(svr.D{v}(:));     
            svr.D{v}=single(svr.D{v}>svr.Alg.RemoveSlicesFactor*meaD);
        end
        if ismember(1,computeRes) || ismember(-1,computeRes)
            y=svr.y{v};
            if useGPU;y=gpuArray(y);end
            xc=xc-y;
            if ismember(-1,computeRes);xc=-xc;end
        end
        
        %ASSIGNMENT
        if isa(svr.yy{v},'gpuArray');svr.yy{v}=xc;else svr.yy{v}=gather(xc);end
    end
end
