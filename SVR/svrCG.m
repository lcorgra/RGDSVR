function svr=svrCG(svr,nIt,tol)

%SVRCG   Performs a CG-based pseudoinverse reconstruction for SVR
%   SVR=SVRCG(SVR,{NIT},{TOL})
%   * SVR is a svr structure
%   * {NIT} is the maximum number of iterations
%   * {TOL} is the tolerance set for convergence
%   ** SVR is a svr structure after the reconstruction step
%

if nargin<2 || isempty(nIt);nIt=300;end
if nargin<3 || isempty(tol);tol=1e-2;end

if isfield(svr,'xz') && ~isempty(svr.xz)
    svr.yy=cell(1,svr.NV);
    svr.xx=svr.xz;
    svr=svrEncode(svr,-1); 
else
    svr.yy=svr.y;
end
svr=svrDecode(svr);
r=svr.xx;

if isfield(svr,'xz') && ~isempty(svr.xz);x=svr.x-svr.xz;else x=svr.x;end
svr.xx=x;
svr=svrDecode(svrEncode(svr));
EHE=svr.xx;
ti=svr.Alg.Ti;
if ~isempty(svr.F)
    for g=1:length(svr.F)
        if ti(g)~=0
            if svr.Alg.RegFracOrd(g)==0 && svr.Alg.Windowing(2)>0;EHE=EHE+bsxfun(@times,real(filtering(x,svr.F{g},1)),min(1./svr.Hx{1},1000));
            else EHE=EHE+real(filtering(x,svr.F{g},1));
            end
        end
    end
else
    EHE=EHE+svr.Alg.Lambda*x;
end

r=r-EHE;
z=r;
    
p=z;
zr=real(conj(z).*r);
rsold=sum(zr(:));
if rsold<1e-10
    svr.x=z;
    if isfield(svr,'xz') && ~isempty(svr.xz);svr.x=svr.x+svr.xz;end
    svr.x=abs(svr.x);
    return
end  

err=inf;
for n=1:nIt   
    svr.xx=p;
    svr=svrDecode(svrEncode(svr));
    EHE=svr.xx;
    if ~isempty(svr.F)
        for g=1:length(svr.F)
            if ti(g)~=0
                if svr.Alg.RegFracOrd(g)==0 && svr.Alg.Windowing(2)>0;EHE=EHE+bsxfun(@times,real(filtering(p,svr.F{g},1)),min(1./svr.Hx{1},1000));
                else EHE=EHE+real(filtering(p,svr.F{g},1));
                end
            end
        end
    else
        EHE=EHE+svr.Alg.Lambda*p;
    end
    g=rsold./sum(real(conj(p(:)).*EHE(:)));
    xup=g*p;
    x=x+xup;
              
    r=r-g*EHE;
    z=r;
        
    zr=real(conj(z).*r);
    rs=sum(zr(:));
    d=rs/rsold;

    p=z+d*p;
    rsold=rs;
    if err<tol || abs(rs)<1e-6;break;end
end

svr.x=x;
if isfield(svr,'xz') && ~isempty(svr.xz);svr.x=svr.x+svr.xz;end
svr.x=abs(svr.x);
