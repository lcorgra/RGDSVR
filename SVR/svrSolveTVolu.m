function [svr,conv]=svrSolveTVolu(svr)

%SVRSOLVETVOLU   Solve for a rigid transform T on a per-volume level
%   [SVR,CONV]=SVRSOLVETVOLU(SVR)
%   * SVR is a svr structure
%   ** SVR is a svr structure after an iteration of solver
%   ** CONV is a flag to indicate convergence
%

gat=1;%1 to save GPU memory
gpu=useGPU;
NT=size(svr.TV);

a=[1 2 3 1 1 2 1 2 3 1 2 3 1 2 3 4 4 5 4 5 6;
   1 2 3 2 3 3 4 4 4 5 5 5 6 6 6 5 6 6 4 5 6];
NHe=size(a,2);
dHe=single(zeros([NHe NT(5)]));
dH=single(zeros(NT([6 5])));dHEff=dH;
E=single(zeros(NT(1:5)));
if ~isfield(svr,'convT');convT=single(false(NT(1:5)));else convT=svr.convT;end

multA=1.2;multB=2;%Factors to divide/multiply the weight that regularizes the Hessian matrix when E(end)<E(end-1)
     
winit=svr.Alg.Winit(1);
if ~isfield(svr,'w');svr.w=winit*ones(NT(1:5));end
svr.w=min(svr.w,winit);
flagw=zeros(NT(1:5));
perm=[5 6 1:4];
Eprev=E;
dHe(:)=0;dH(:)=0;
vT=find(~convT)';

conv=0;
while ~conv
    for v=vT         
        xT=svr.x;    
        [et,etg]=precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(svr.TV,v,5),1,1,[],1,svr.cGrid);
        [xT,xB]=sincRigidTransformQuick(xT,et,1,svr.FT,[],gat);    
        svr.xx=real(xT);
        svr=svrEncode(svr,2,v,4);%Note here we also compute the coefficients to discard those slices that project to the background (second parameter=2)
        x=svr.yy{v};y=svr.y{v};
        J=sincRigidTransformGradientQuick(xB,et,etg,svr.FT,[],gat);xB=[];     
        for m=1:NT(6)
            if gpu;J{m}=gpuArray(J{m});end
            svr.xx=real(J{m});
            svr=svrEncode(svr,0,v,4);
            J{m}=svr.yy{v};
        end    
        [Eprev(v),dH(:,v),dHe(:,v)]=computeMetricDerivativeHessianRigid(x,y,'MSD',svr.G{1}{v},[],J);J=[];
    end

    E=Eprev;
    MHe=eye(6,'single');
    flagw(:)=0;    
    fina=0;
    while fina==0
        vT=find(~convT)';
        for s=vT
            for k=1:NHe
                if a(1,k)==a(2,k)              
                    MHe(a(1,k),a(2,k))=(1+svr.w(s))*dHe(k,s)+1e-9;%1e-9 serves to stabilize
                else
                    MHe(a(1,k),a(2,k))=dHe(k,s);MHe(a(2,k),a(1,k))=dHe(k,s);
                end              
            end   
            dHEff(:,s)=-winit*single(double(MHe)\double(dH(:,s)))/svr.w(s);
        end     
        dHEff(:,svr.w>1e10 | convT)=0;
        permH=[3 4 5 6 2 1];
        Tupr=permute(dHEff,permH);
        Tup=svr.TV+Tupr;
        Tup=restrictTransform(Tup);    

        vT=find(~convT(:) & flagw(:)~=2)';
        for v=vT
            xT=svr.x;        
            et=precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(Tup,v,5),1,0,[],1,svr.cGrid);                
            xT=sincRigidTransformQuick(xT,et,1,svr.FT);       
            svr.xx=real(xT);
            svr=svrEncode(svr,0,v,4);
            x=svr.yy{v};y=svr.y{v};
            E(v)=computeMetricDerivativeHessianRigid(x,y,'MSD',svr.G{1}{v},[]);
        end
        E(svr.w>1e10 & ~convT)=Eprev(svr.w>1e10 & ~convT);
        flagw(E<=Eprev)=2;               

        if any(flagw==1 | flagw==0)  
            svr.w(E>Eprev & ~convT)=svr.w(E>Eprev & ~convT)*multB;
        else   
            svr.w(~convT)=svr.w(~convT)/multA;
            svr.w(svr.w<1e-8)=multA*svr.w(svr.w<1e-8);%To avoid numeric instabilities
            %Tupr=bsxfun(@minus,Tupr,mean(Tupr,5));%This gives problems with regularization, but disabling it may provoke drifting...
            svr.TV=svr.TV+Tupr;
            svr.TV=restrictTransform(svr.TV);
            fina=2;
            traMax=abs(permute(dynInd(Tupr,1:3,6),perm));
            rotMax=convertRotation(abs(permute(dynInd(Tupr,4:6,6),perm)),'rad','deg');
            traLim=0.16;rotLim=0.08;
            traLim=traLim*svr.Alg.ConvL;rotLim=rotLim*svr.Alg.ConvL;
            if max(traMax(:))>traLim || max(rotMax(:))>rotLim;fina=1;end
            convP=max(traMax,[],2)<traLim & max(rotMax,[],2)<rotLim;            
            convT(convP)=convT(convP)+1;
            convT(~convP)=0;
            conv=all(convT>=svr.Alg.NconvT);
            if svr.Alg.Debug
                fprintf('Energy before: %0.6g / Energy after: %0.6g\n',sum(Eprev),sum(E));
                fprintf('Maximum change in translation (vox): ');fprintf('%0.3f ',max(traMax,[],1));
                fprintf('/ Maximum change in rotation (deg): ');fprintf('%0.3f ',max(rotMax,[],1));fprintf('\n');
                fprintf('Not converged motion states: %d of %d\n',NT(5)-sum(single(convT>=svr.Alg.NconvT)),NT(5));
            end        
            if conv;svr=rmfield(svr,'w');end
        end 
    end        
end
svr.convT=convT;
if conv;svr=rmfield(svr,'convT');end