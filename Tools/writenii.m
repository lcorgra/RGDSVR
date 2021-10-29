function writenii(fil,x,suff,MS,MT,datima)

%WRITENII   Writes a set of nii files and potentially associated metafila
%   WRITENII(FIL,X,{SUFF},{MS},{MT},{DATIMA})
%   * FIL indicates the path and name of the file
%   * X is the fila to write
%   * {SUFF} is a list of suffes of the files to be written, it defaults
%   to no suff if x is an array and not a cell
%   * {MS} is the spacing of the fila, defaults to 1
%   * {MT} is the orientation of the fila, defaults to identity
%   * {DATIMA} is potentially associated metadata
%

if nargin<3 || isempty(suff);suff={''};end
if nargin<6;datima=[];end
if ~iscell(x);x={x};end
if ~iscell(suff);suff={suff};end
if ~iscell(datima) && ~isempty(datima);datima={datima};end
Nd=length(datima);

%IF FOLDER NOT EXISTING, WE CREATE IT
fol=fileparts(fil);
if ~exist(fol,'dir');mkdir(fol);end

%IF EXTENSION IS ENTERED, WE REMOVE IT
fil=removeExtension(fil,{'.nii','.gz'});

%DEFAULT STRUCTURES
N=length(suff);
if nargin<5 || isempty(MT)
    if nargin<4 || isempty(MS);[MS{1:N}]=deal(ones(1,3));end 
    MT=cellfun(@(x)blkdiag(diag(x),1),MS,'UniformOutput',false);
else
    if isempty(MS);[~,~,MS]=factorizeHomogeneousMatrix(MT);MS=diagm(MS);MS=MS(1:3);end
end

if ~iscell(MS);MS={MS};end
if ~iscell(MT);MT={MT};end
NIn=[length(suff) length(x) length(MS) length(MT)];
assert(length(unique(NIn))==1,'Size of input cells must be the same. Currently it is: %d %d %d %d',length(suff),length(x),length(MS),length(MT));

%WRITE IMAGES
for n=1:N    
    if ~isempty(suff{n});suff{n}=strcat('_',suff{n});end
    M=size(x{n});M(end+1:12)=1;
    x{n}=reshape(single(x{n}),[M(1:3) prod(M(4:12))]);    
    if any(imag(x{n}(:)))
        p=gather(angle(x{n}));       
        niftiIm=make_nii(p,MS{n}(1:3));
        niftiIm.hdr.hist.srow_x=MT{n}(1,:);niftiIm.hdr.hist.srow_y=MT{n}(2,:);niftiIm.hdr.hist.srow_z=MT{n}(3,:);niftiIm.hdr.hist.sform_code=1;         
        if length(MS{n})>=4;niftiIm.hdr.dime.pixdim(5)=MS{n}(4);end 
        save_nii(niftiIm,strcat(fil,suff{n},'Ph.nii'));
        x{n}=abs(x{n});
    end
    x{n}=gather(real(x{n}));    
    niftiIm=make_nii(x{n},MS{n}(1:3));
    niftiIm.hdr.hist.srow_x=MT{n}(1,:);niftiIm.hdr.hist.srow_y=MT{n}(2,:);niftiIm.hdr.hist.srow_z=MT{n}(3,:);niftiIm.hdr.hist.sform_code=1;            
    if length(MS{n})>=4;niftiIm.hdr.dime.pixdim(5)=MS{n}(4);end
    save_nii(niftiIm,strcat(fil,suff{n},'.nii'));
    if n<=Nd && ~isempty(datima{n})
        dat=datima{n};
        save(strcat(fil,suff{n},'.mat'),'dat');
    end
end
