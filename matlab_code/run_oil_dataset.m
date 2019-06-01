X = oil_data();
%display('Data:');
%display(X);

subplot(1,2,1);
scatter(X(1,:),X(2,:));  %visualization
title('Dimensions 1-2 of data');

axis equal;

display('Empirical covariance of initial data:');
cov(X')                  %(covariance is close to identity)


display('Size of X:');
size(X)                         % check the size of X (1000
                                % points in 13 dimensions)

tic
[ngspace,projdata,signalspace] = NGCA(X,[]);  % apply NGCA with
                                              % defaults parameters
					      % (in particular,
                                              % searches for 2
                                              % non-Gaussian dimensions)
timeElapsed = toc;
display(timeElapsed)

subplot(1,2,2);
						 
scatter(projdata(1,:),projdata(2,:));    
title('Retrieved data');
axis equal;
