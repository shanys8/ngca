function [ ngmatrix, projdata, signalmatrix] = NGCA(data,parameters,nfun)
%
% THIS CODE IS IN BETA STATE - please send bugs and comments to gilles.blanchard@gmail.com
%
% [ ngmatrix, projdata, signalmatrix] = NGCA(data,parameters,nfun)
%
%
% Multi-index FastICA using the "NGCA" methodology
%
%
%
% data                   :   data vector (dimension x nb_examples)
%
% parameters.maxsigmasq  :   maximum sigma^2 for gaussian part of Gausspoly (default 5)
%           .renorm      :   Use renormalization (0: no; 
%                                                 1: using estimated stdev (default))
%           .maxfreq     :   upper bound on the frequency (for Fourier) (default 4)
%           .maxfactor   :   upper bound on the factor (for tanh) (default 5)
%           .gausspoly   :   1 (default) use gausspoly index, 0 don't
%           .fourier     :   1 (default) use sine/cosine index, 0 don't
%           .tanh        :   1 (default) use tanh index, 0 don't
%           .threshold   :   discard vectors that have norm below this threshold. (default 1.6)
%           .nbiter      :   nb internal iterations for fastICA (default 10)
%           .nbng        :   nb components to search for        (default 2)
%           .projection  :   0  : coordinates or orthogonal projection of data on NG subspace in original space
%                            1 (default): coordinates of orthogonal projection on NG subspace in whitened space
%                            2 : coordinates of orthogonal projection on signal subspace in original space
%              NOTE: policies 0 and 1 only differ from an invertible linear transform
%           .nbatch      : nb of vectors for batch-FastICA, default 1
% [nfun]                 : nb of functions to combine for each index (default 1001)
%
% The data is first 'whitened' before further processing.
%
% The 'variance' is estimated for each vector for renormalization (for renorm=1).
% Selection of directions is realized using fastICA.
%
% Out: 
%      ngmatrix : orth. basis of NG space in original space.
%      projdata : coordinates of projected data following projection policy
%      signalmatrix : orth. basis of estimated signal space in original space.
%
% GB 07/08/06

% offsets to avoid degeneracies

minifreq = .05;
minifactor = .05;
minisigmasq = .5;

if ~isfield(parameters,'maxsigmasq')
  parameters.maxsigmasq = 5;
end;

if ~isfield(parameters,'renorm')
  parameters.renorm = 1;
end;

if ~isfield(parameters,'maxfreq')
  parameters.maxfreq = 4;
end;

if ~isfield(parameters,'maxfactor')
  parameters.maxfactor = 5;
end;

if ~isfield(parameters,'gausspoly')
  parameters.gausspoly = 1;
end;

if ~isfield(parameters,'fourier')
  parameters.fourier = 1;
end;

if ~isfield(parameters,'tanh')
  parameters.tanh = 1;
end;

if ~isfield(parameters,'nbiter')
  parameters.nbiter = 10;
end;

if ~isfield(parameters,'nbng')
  parameters.nbng = 3;
end;

if ~isfield(parameters,'threshold')
  parameters.threshold = 1.6;
end;

if ~isfield(parameters,'sphered_proj')
  parameters.sphered_proj = 1;
end;

if ~isfield(parameters,'nbatch')
  parameters.nbatch = 1;
end;

nbatch = parameters.nbatch;

  opts.disp=0;  % for function eigs later

if (nargin<3)
  nfun=1001;   % 1001 is just to yield dimension matching errors if something is buggy
end;

if (nargin<4)
  makeplot=0;
end;

x = data;

meanx = mean(x,2);

x=x-repmat(meanx,1,size(x,2));   %center data

[d,n]=size(x);

sigmaxx = x*x'/n;
  
[ev,ed] = eig(sigmaxx);

x = ev*diag(1./sqrt(diag(ed)))*ev'*x;  %we 'whiten' the data

% This is the Gauss-poly part.


if (parameters.gausspoly == 0)
  betav1=[];
  V1=[];
else
  sigmasq = (1:nfun)/nfun*(parameters.maxsigmasq-minisigmasq)+minisigmasq;

  y = randn(d,nfun,nbatch); 
  y = y./repmat(sqrt(sum(y.^2,1)), [d 1 1]);                 % y is now uniformly distributed on the sphere
  
  for i=1:(parameters.nbiter-1)
    for j=1:nbatch
      betav = NGCA_Gausspow3(x,y(:,:,j),sigmasq);
      for k=1:(j-1)
	betav = betav - repmat( sum(betav.*y(:,:,k),1), [d 1]).*y(:,:,k); 
	           % subtract projections on already obtained components for batchmode
      end;
      y(:,:,j) = betav./repmat(sqrt(sum(betav.^2)),[d 1]); 
	  %renormalize beta's and use them for next directions (fastICA method);
    end;
  end;
   

  switch (parameters.renorm)
      case 1
	for j=1:nbatch
	  [betav1(:,((j-1)*nfun+1):(j*nfun)),V1(((j-1)*nfun+1):(j*nfun))] = NGCA_Gausspow3(x,y(:,:,j),sigmasq);
	end;
	betav1 = betav1./repmat(sqrt(V1),[size(betav1,1) 1]);
      case 0
	betav1 = NGCA_Gausspow3(x,y,sigmasq);
      case 2
	countnr = 1;
	betav1 = NGCA_Gausspow3(x,y,sigmasq);
	betav1 = betav1./repmat(qtiles(1:nfun),[size(betav1,1) 1]);
  end;
    
end

% This is the Fourier part.

if (parameters.fourier == 0 )
  betav2=[];
  V2=[];
  betav2b=[];
  V2b=[];
else
  freq = (1:nfun)/nfun*(parameters.maxfreq - minifreq) + minifreq;
 
  y = randn(d,nfun,nbatch); 
  y = y./repmat(sqrt(sum(y.^2,1)), [d 1 1]);                 % y is now uniform distributed on the sphere
  
  for i=1:(parameters.nbiter-1)
    for j=1:nbatch
      betav = NGCA_cosine(x,y(:,:,j),freq);
      for k=1:(j-1)
	betav = betav - repmat( sum(betav.*y(:,:,k),1), [d 1]).*y(:,:,k); 
	           % subtract projections on already obtained components for batchmode
      end;
      y(:,:,j) = betav./repmat(sqrt(sum(betav.^2)),[d 1]); 
	  %renormalize beta's and use them for next directions (fastICA method);
    end;
  end;

  switch (parameters.renorm)
    case 1
      for j=1:nbatch
	[betav2(:,((j-1)*nfun+1):(j*nfun)),V2(((j-1)*nfun+1):(j*nfun))] = NGCA_cosine(x,y(:,:,j),freq);
      end;
    case 0  
      betav2 = NGCA_cosine(x,y,freq);
    case 2
      betav2 = NGCA_cosine(x,y,freq);
      betav2 = betav2./repmat(qtiles((countnr*nfun+1):((countnr+1)*nfun)),[size(betav2,1) 1]);
      countnr = countnr + 1;
  end;

  
  y = randn(d,nfun,nbatch); 
  y = y./repmat(sqrt(sum(y.^2,1)), [d 1 1]);                 % y is now uniform distributed on the sphere
  
  for i=1:(parameters.nbiter-1)
    for j=1:nbatch
      betav = NGCA_sine(x,y(:,:,j),freq);
      for k=1:(j-1)
	betav = betav - repmat( sum(betav.*y(:,:,k),1), [d 1]).*y(:,:,k); 
	           % subtract projections on already obtained components for batchmode
      end;
      y(:,:,j) = betav./repmat(sqrt(sum(betav.^2)),[d 1]); 
	  %renormalize beta's and use them for next directions (fastICA method);
    end;
  end;

  switch (parameters.renorm)
    case 1
      for j=1:nbatch
	[betav2b(:,((j-1)*nfun+1):(j*nfun)),V2b(((j-1)*nfun+1):(j*nfun))] = NGCA_sine(x,y(:,:,j),freq);
      end;
    case 0
      betav2b = NGCA_sine(x,y,freq);
    case 2
      betav2b = NGCA_sine(x,y,freq);
      betav2b = betav2b./repmat(qtiles((countnr*nfun+1):((countnr+1)*nfun)),[size(betav2b,1) 1]);
      countnr = countnr + 1;
  end;

end

% This is the tanh part.

if (parameters.tanh == 0)
  betav3=[];
  V3=[];
else
  factor = (1:nfun)/nfun*(parameters.maxfactor - minifactor) + minifactor;
  
  y = randn(d,nfun,nbatch); 
  y = y./repmat(sqrt(sum(y.^2,1)), [d 1 1]);                 % y is now uniform distributed on the sphere
  
  for i=1:(parameters.nbiter-1)
    for j=1:nbatch
      betav = NGCA_tanh(x,y(:,:,j),factor);
      for k=1:(j-1)
	betav = betav - repmat( sum(betav.*y(:,:,k),1), [d 1]).*y(:,:,k); 
	           % subtract projections on already obtained components for batchmode
      end;
      y(:,:,j) = betav./repmat(sqrt(sum(betav.^2)),[d 1]); 
	  %renormalize beta's and use them for next directions (fastICA method);
    end;
  end;

  switch (parameters.renorm)
    case 1
      for j=1:nbatch
	[betav3(:,((j-1)*nfun+1):(j*nfun)),V3(((j-1)*nfun+1):(j*nfun))] = NGCA_tanh(x,y(:,:,j),factor);
      end;
    case 0
      betav3 = NGCA_tanh(x,y,factor);
    case 2
      betav3 = NGCA_tanh(x,y,factor);
      betav3 = betav3./repmat(qtiles((countnr*nfun+1):((countnr+1)*nfun)),[size(betav3,1) 1]);
      countnr = countnr + 1;
  end;
  
end;

betav = [betav1 betav2 betav2b betav3];

normbetav = sqrt(sum(betav.^2,1));

betavselect = betav(:,normbetav>parameters.threshold/sqrt(n));

[eigvec1,eigval1] = eigs(betavselect*betavselect',parameters.nbng,'LM',opts);  % PCA in sphered space

eigvec = ev*diag(1./sqrt(diag(ed)))*ev'*eigvec1; %orthogonal of Gaussian subspace in original space

signal_space = ev*diag(sqrt(diag(ed)))*ev'*eigvec1; %signal subspace in original space

[eigvec, R] = qr(eigvec,0); % we find an orthonormal basis for the span of eigvec

[signal_space, R] = qr(signal_space,0); % idem for the signal space

ngmatrix = eigvec;

signalmatrix = signal_space;

% eigvec = orth(eigvec); % alternative code: uses SVD. See help orth . We prefer the above qr method
                         % because we know the vectors are independent.

switch (parameters.sphered_proj)
  case 1
    projdata = eigvec1'*x; %orth. projection of (centered and whitened) data on NG subspace
    projmatrix = eigvec1'*ev*diag(1./sqrt(diag(ed)))*ev'; 
    % oblique projection matrix (in original space) to project new data
  case 0
    projdata = eigvec'*data; % orth. projection in original space.
    projmatrix = eigvec';
 case 2 
    projdata = signalmatrix'*data; %orth. projection on signal subspace in original space.
end;




evect = eigvec;

if false
  projdata = evect'*data; % data projected on the directions.
end;

