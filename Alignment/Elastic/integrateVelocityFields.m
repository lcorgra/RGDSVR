function phiinv=integrateVelocityFields(vt,sp,G,ups)

%INTEGRATEVELOCITYFIELDS   Integrates velocity fields to obtain
%diffeomorphisms (given as displacement fields)
%   PHI=INTEGRATEVELOCITYFIELDS(VT,SP,{G},{UPS})
%   * VT are the velocity fields at different instants in time (in Fourier)
%   * {SP} is the spacing of the spatial grid
%   * {G} is the gradient type, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions
%   * {UPS} is a padding factor for circular convolutions
%   ** PHIINV is the backward map (in same space as vt)
% 

gpu=isa(vt,'gpuArray');
ND=length(sp);
N=size(vt);N(end+1:ND+3)=1;
if nargin<3 || isempty(G);G='1stFiniteDiscreteCentered';end
if nargin<4 || isempty(ups);ups=2;end
if isscalar(ups);ups=ups*ones(1,ND);end

NT=N(ND+3);
NTV=1:NT;
dt=1/length(NTV);
NX=N(1:ND);
if ~any(ups==0)%We assume 1stFiniteDiscreteCentered and spatial computation      
    G=buildGradientOperator(G,N,sp,gpu);
    M=round(NX.*ups);ups=M./NX;
    if NT>1 || nargout>=2;F=buildMapSpace(vt,1,M,NX);FH=buildMapSpace(vt,0,M,NX);end
end

for n=NTV   
    vn=-dynInd(vt,n,ND+3);
    if n>NTV(1)
        vnc=permute(vn,[1:ND ND+2 ND+1]);       
        if any(ups==0)
            phiu=deformationGradientTensorSpace(phiinv,sp);
        else
            phiu=deformationGradientTensor(phiinv,sp,G,1);        
            phiu=mapSpace(phiu,0,F,FH);vnc=mapSpace(vnc,0,F,FH);
        end
        phiu=sum(bsxfun(@times,phiu,vnc),ND+2);
        if ~any(ups==0);phiu=mapSpace(phiu,1,F,FH);end
        u=(phiu+vn)*dt;
        %if ~any(ups==0)
        %    u=mapSpace(u,0,F,FH);
        %    phiinv=mapSpace(phiinv,0,F,FH);
        %end
        %phiinv=elasticTransform(phiinv,u,sp);
        %if ~any(ups==0);phiinv=mapSpace(phiinv,1,F,FH);end
        phiinv=phiinv+u;
    else
        u=vn*dt;
        %if ~any(ups==0);u=mapSpace(u,0,F,FH);end        
        phiinv=u;
        %if ~any(ups==0);phiinv=mapSpace(phiinv,1,F,FH);end
    end
end
