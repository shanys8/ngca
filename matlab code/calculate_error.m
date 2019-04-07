function error = calculate_error(u1, u2);
error = subspace(u1, u2);
%error = norm(u1 * transpose(u1) - u2 * transpose(u2));