function [flagw,w,Eprev,DT,fina,convT,conv]=finalizeConvergenceControl(nIt,flagw,E,Eprev,Einit,convT,NconvT,w,Tup,DT,fina,NXT,debug,parGrouping,spGridX,convL,baseConvLevel,NT,modw,ups)

%FINALIZECONVERGENCECONTROL   Updates variables related with convergence 
%control for elastic motion estimation in SVR
%   [FLAGW,W,EPREV,DP,FINA,CONVT,CONV]=FINALIZECONVERGENCECONTROL(NIT,FLAGW,E,EPREV,EINIT,CONVT,NCONVT,W,TUP,DT,FINA,NXT,DEBUG,PARGROUPING,SPGRIDX,CONVL,BASECONVLEVEL,NT,MODW,UPS);
%   * NIT is the total number of iterations
%   * FLAGW are flags used to indicate partial convergence for different
%   motion states
%   * E is current energy
%   * EPREV is previous energy
%   * EINIT is starting energy
%   * CONVT indicates partial convergence status for different motion
%   states
%   * NCONVT is the required number of iterations with motion updates below
%   a certain threshold for establishing convergence
%   * W are the weights in the update rule
%   * TUP are the candidate motion parameters
%   * DT are the motion parameters
%   * FINA is a flag indicating line search status
%   * NXT is the temporal grid size
%   * DEBUG serves to show information
%   * PARGROUPING indicates whether convergence should be established using
%   a group of transforms (if a vector) or globally (if empty)
%   * SPGRIDX is the spatial grid resolution
%   * CONVL serves to accelerate convergence to set it as a factor of 
%   BASECONVLEVEL
%   * BASECONVLEVEL is a level for convergence as a percentage of the voxel
%   size
%   * NT is the number of transforms
%   * MODW indicates whether w has been modified
%   * UPS is the upsampling factor for circular convolution, merely to
%   indicate if velocity fiels are in space or spectrum
%   ** FLAGW are flags used to indicate partial convergence for different
%   motion states
%   ** W are the weights in the update rule
%   ** EPREV is the updated previous energy
%   ** DT are the updated motion parameters
%   ** FINA is a flag indicating line search status
%   ** CONVT indicates partial convergence status for different motion
%   states
%   ** CONV indicates global convergence
%

E(w>=1e4)=Eprev(w>=1e4);
if debug;fprintf('Energy before: %0.6g / Energy after: %0.6g\n',sum(Eprev),sum(E));end

%multA=1.2;multB=2;%Factors to divide/multiply the weight that regularizes the Hessian matrix when E(end)<E(end-1)
multA=2;multB=5;%Factors to divide/multiply the weight that regularizes the Hessian matrix when E(end)<E(end-1)
conv=0;

Eu=E;Einitu=Einit;Eprevu=Eprev;        
if ~isempty(parGrouping)
    Eu=repelem(cellfun(@sum,mat2cell(Eu(:),parGrouping)),parGrouping);Eu=Eu(:)';
    Einitu=repelem(cellfun(@sum,mat2cell(Einitu(:),parGrouping)),parGrouping);Einitu=Einitu(:)';
    Eprevu=repelem(cellfun(@sum,mat2cell(Eprevu(:),parGrouping)),parGrouping);Eprevu=Eprevu(:)';
end

if ismember(nIt,1)
    flagw(flagw==1 & Eu<=Einitu)=2;    
    flagw(flagw==0 & Eu==Einitu)=1;    
else                
    flagw(Eu<=Eprevu)=2;           
end
convTT=(convT>=NconvT);
if any(flagw~=2)
    w(Eu>Eprevu & ~convTT & flagw==1)=w(Eu>Eprevu & ~convTT & flagw==1)*multB;%Normal case
    w(Eu>Eprevu & ~convTT & flagw==0 & ~modw)=w(Eu>Eprevu & ~convTT & flagw==0 & ~modw)*multA;%Case with full search, we have reached the minimum
    flagw((Eu>=Eprevu | convTT) & flagw==0)=1;
    w(Eu<=Eprevu & ~convTT & flagw==0)=w(Eu<=Eprevu & ~convTT & flagw==0)/multA;%Case with full search, we keep on searching

    Eprev(~convTT & flagw==0)=E(~convTT & flagw==0);
else   
    w(~convTT)=w(~convTT)/multA;
    w(w<1e-8)=multA*w(w<1e-8);%To avoid numeric instabilities 
    Tupr=Tup-DT;
    DT=Tup;     
    fina=2;    
    if ups>0;parMax=mapSpace(Tupr,0,NXT,NXT);else parMax=Tupr;end
    parMax=multDimMax(sqrt(sum(parMax.*parMax,4)),1:3);
    parMax=parMax(:);
    if debug;fprintf('Maximum change in parameters (mm): ');fprintf('%0.3f ',max(parMax));end
    if ~isempty(parGrouping);parMax=repelem(cellfun(@max,mat2cell(parMax,parGrouping)),parGrouping);end
    parLim=(baseConvLevel*max(spGridX))*convL;
    convT(parMax<parLim)=convT(parMax<parLim)+1;
    convT(parMax>=parLim)=0;
    if any(convT(:)<NconvT);fina=1;end
    if debug;fprintf('Not converged motion states: %d of %d\n',sum(single(convT<NconvT)),NT);end
    conv=all(convT) && nIt<=NconvT;
end
