function [beta,V] = NGCA_Gausspow3(x,y,sigmasq)
%
%
% Apply NGCA (=one step of fastICA) to data x, directions y
% For Gausspoly fns; sigmasq is vector of sq. widths for the Gauss part.
%
%
% GB 01/12/04

deg = 3; % might be changed later

[dimx,nx] = size(x);
[dimy,ny] = size(y);
[dimssq, nssq] = size(sigmasq);

if (dimx ~= dimy)
  disp('X and Y do not have the same first dimension!');
  return;
end;

if (dimssq >1)
 disp('sigmasq must be a row vector');
 return;
end;

if (nssq ~= ny)
  disp('Second dimension of sigmasq and y must match');
  return;
end;

y = y./repmat(sqrt(sum(y.^2,1)),[dimy 1]); % for security: normalize y

xdoty = x'*y;

inexp = (xdoty.^2)./repmat(sigmasq,[nx 1]);

expart = exp(-inexp/2);

H = (xdoty.^deg).*expart;

DH = (xdoty.^(deg-1)).*(deg-inexp).*expart;

beta = y.*repmat(mean(DH,1),[dimy 1]) - x*H/nx;


if (nargout>1)
  V = mean(DH.^2,1) + sum(x.^2,1)*(H.^2)/nx - 2*mean(xdoty.*H.*DH,1) - sum(beta.^2,1);
end;

