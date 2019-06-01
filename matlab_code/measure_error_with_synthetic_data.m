[NG_subspace, X] = generate_random_data(100, 10, 3);


[ngspace,projdata,signalspace] = NGCA(X,[]);  % apply NGCA with
                                              % defaults parameters
					      % (in particular,
                                              % searches for 3
                                              % non-Gaussian dimensions)

Error = calculate_error(NG_subspace, ngspace);
      
display('Error:');
Error

