function svr=svrDecode(svr,sep,vU,useT)

%SVRDECODE   Decoding operator for SVR
%   SVR=SVRDECODE(SVR,{SEP},{VU},{USET})
%   * SVR is a svr structure
%   * {SEP} indicates whether to return separate views
%   * {VU} serves to decode only a specific set of views
%   * {USET} serves to indicate that the decoding includes the transform
%   ** SVR is a svr structure after decoding
%

if nargin<2 || isempty(sep);sep=0;end%Sep=0, full decoding / Sep=1 full decoding without adding at the end / Sep=2 no weighting adding at the end
if nargin<3 || isempty(vU);vU=[];end
if nargin<4 || isempty(useT);useT=5;end
gpu=useGPU;

mirr=2*ones(1,3);

if svr.UseInterleave==1
    TPI=svr.TI;
    DPIH=svr.DIH;
    MPI=svr.MInte;
    slPerPI=svr.slPerInte;
    NPI=svr.I;
else
    TPI=svr.TP;
    DPIH=svr.DPH;
    MPI=svr.MPack;
    slPerPI=svr.slPerPack;
    NPI=svr.P;
end

%RECONSTRUCTED DATA
svr.xx=zeros(svr.NX,'like',svr.x);
NV=svr.NV;
if length(vU)==1;NV=1;end
if sep==1;svr.xx=repmat(svr.xx,[ones(1,3) NV]);end

for v=1:svr.NV     
    if length(vU)<1 || v==vU(1)
        x=svr.yy{v};
        if gpu;x=gpuArray(x);end

        if sep<2            
            %WEIGHTS  
            x=bsxfun(@times,x,svr.W{v});

            %WINDOWING
            if svr.Alg.Windowing(1)>0
                H=svr.H{1}{svr.NYFullInd(v)};
                if gpu;H=gpuArray(H);end
                x=bsxfun(@times,x,H);
            end
        end
        
        xc=zeros(svr.NX,'like',x);
        useP=(any(TPI{v}(:)~=0) && useT>2) || (any(DPIH{v}(:)~=0) && useT>1);
        for p=1:NPI(v)
            if length(vU)<2 || p==vU(2)
                NS=slPerPI{v}(p);
                useS=any(svr.TE{v}{p}(:)~=0) && useT>0;
                if ~useS;NS=1;end
                if useS;useP=1;end
                        
                %PER-PACK MASKING
                if useP;xp=bsxfun(@times,x,dynInd(MPI{v},p,5));else xp=x;end

                xb=zeros(svr.NX,'like',xp);
                for s=1:NS
                    if length(vU)<3 || s==vU(3)
                        xs=xp;                                                
                        %PER-EXCITATION MASKING
                        if useS;xs=bsxfun(@times,xs,dynInd(svr.MSlices{v}{p},s,6));end                                                                                              

                        %SLICE DELTA-SAMPLING
                        dsl=svr.id(v,3);
                        xs=ifold(xs,dsl,svr.NXY(v,dsl),svr.NY(v,dsl),[],[],1);                       

                        %RESAMPLING
                        xs=real(resampling(xs,svr.NXY(v,:),[],mirr));
                        
                        %SLICE PROFILE
                        xs=filtering(xs,svr.SlPr{v},1);

                        %FILL ARRAY                        
                        NNXX=size(xp);NNXX(end+1:4)=1;    
                        xa=zeros([svr.NX NNXX(4:end)],'like',xs);
                        xb=dynInd(xa,svr.vRec{v},1:3,dynInd(xs,svr.vAcq{v},1:3));
                    end
                end;xa=[];xs=[];

                %PER-PACK TRANSFORM
                if useT>1;xb=real(sincRigidTransformQuick(xb,precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(TPI{v},p,5),0,0,[],1,svr.cGrid),0,svr.FT));end            
                if useT>2;xb=elasticTransform(xb,dynInd(DPIH{v},p,5),svr.MSX,svr.Alg.Ups(3));end
                xc=xc+xb;
                if ~useP;break;end
            end            
        end;xb=[];xp=[];

        %PER-VOLUME TRANSFORM   
        if useT>3 && isfield(svr,'DVH');xc=elasticTransform(xc,dynInd(svr.DVH,v,5),svr.MSX,svr.Alg.Ups(3));end
        if useT>4;xc=real(sincRigidTransformQuick(xc,precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(svr.TV,v,5),0,0,[],1,svr.cGrid),0,svr.FT));end      
        if ismember(sep,[0 2]);svr.xx=svr.xx+xc;else;svr.xx=dynInd(svr.xx,v,4,xc);end
    end
end
