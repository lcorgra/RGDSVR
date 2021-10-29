function [net,xz]=svrDD(svr,factEpochs)    

%SVRDD   Perfoms the DD fit
%   [NET,XZ]=SVRDD(SVR,{FACTEPOCHS})
%   * SVR is a svr structure
%   * {FACTEPOCHS} is a factor that serves to modify the baseline number of 
%   epochs, it defaults to 1
%   ** NET is the updated network
%   ** XZ is the updated projection to natural images
%

if nargin<2 || isempty(factEpochs);factEpochs=1;end

gpu=isa(svr.x,'gpuArray');

%DATA TO FIT
maxx=max(svr.x(:));
svr.dd.y=svr.x/maxx;
svr.dd.parFit.y_ast=svr.x/maxx; 
svr.dd.y=gather(permute(svr.dd.y,[svr.dd.parNet.ND+2 svr.dd.parNet.ND+1 1:svr.dd.parNet.ND]));%Samples / Channels / Space
svr.dd.parFit.y_ast=gather(permute(svr.dd.parFit.y_ast,[svr.dd.parNet.ND+2 svr.dd.parNet.ND+1 1:svr.dd.parNet.ND]));%Samples / Channels / Space    

%FITTING
svr.dd.parFit.epochs=svr.dd.parFit.epochs*factEpochs;
tra=cell(svr.df.opt.fit(svr.dd.x,svr.dd.y,svr.dd.net,svr.dd.parFit));
[net,xz]=deal(tra{1:2});    
svr.dd.parFit.epochs=svr.dd.parFit.epochs/factEpochs;


%FOR REGULARIZATION
xz=single(xz);
if gpu;xz=gpuArray(xz);end
xz=permute(xz,[3:svr.dd.parNet.ND+2 2 1]);
xz=xz*maxx;