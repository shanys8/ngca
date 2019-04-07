function [beta,V] = NGCA_tanh(x,y,factor)
%
%
% Apply NGCA (=one step of fastICA) to data x, directions y
% For tanh fns
%
%
% GB 01/12/04

[dimx,nx] = size(x);
[dimy,ny] = size(y);
[dimfact, nfact] = size(factor);

if (dimx ~= dimy)
  disp('X and Y do not have the same first dimension!');
  return;
end;

if (dimfact >1)
 disp('factor must be a row vector');
 return;
end;

if (nfact ~= ny)
  disp('Second dimension of factor and y must match');
  return;
end;

y = (y./repmat(sqrt(sum(y.^2,1)),[dimy 1])).*repmat(factor,[dimy 1]); % for security: renormalize y

xdoty = x'*y;

H = tanh(xdoty);

DH = 1-H.^2;

beta = y.*repmat(mean(DH,1),[dimy 1]) - x*H/nx;


if (nargout>1)
  V = factor.^2.*mean(DH.^2,1) + sum(x.^2,1)*(H.^2)/nx - 2*mean(xdoty.*H.*DH,1) - sum(beta.^2,1);
end;

