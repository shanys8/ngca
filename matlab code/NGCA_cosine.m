function [beta,V] = NGCA_cosine(x,y,freq)
%
%
% Apply NGCA (=one step of fastICA) to data x, directions y
% For cosine fns; freq is vector of frequencies.
%
%
% GB 01/12/04

[dimx,nx] = size(x);
[dimy,ny] = size(y);
[dimfreq, nfreq] = size(freq);

if (dimx ~= dimy)
  disp('X and Y do not have the same first dimension!');
  return;
end;

if (dimfreq >1)
 disp('freq must be a row vector');
 return;
end;

if (nfreq ~= ny)
  disp('Second dimension of freq and y must match');
  return;
end;

y = (y./repmat(sqrt(sum(y.^2,1)),[dimy 1])).*repmat(freq,[dimy 1]) ; % for security: renormalize y

xdoty = x'*y;

Cpart = cos(xdoty);
Spart = sin(xdoty);

H = Cpart;

DH = -Spart;

beta = y.*repmat(mean(DH,1),[dimy 1]) - x*H/nx;

if (nargout>1)
  V = freq.^2.*mean(DH.^2,1) + sum(x.^2,1)*(H.^2)/nx - 2*mean(xdoty.*H.*DH,1) - sum(beta.^2,1);
end;

