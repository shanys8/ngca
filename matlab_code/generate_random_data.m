function [NG, X] = generate_random_data(N, n, d);
%
% Draws n 2D data uniformly in a 4 leaf clover shape
%

threshold = 1e-15;

G = random('normal',0,1, n, n-d);

[Q, R] = qr (G);

[S,V,D]=svd(Q);

rank = trace(V > threshold);

[Q_orth, R] = qr (S);

samples = [];

for j=1:N
	sample = (Q(:,1:(n-d)) * random('normal',0,1, n-d, 1)) + (Q_orth(:,1:d) * (rand(d, 1) - 0.5));
	samples = [samples sample];
end;

NG = Q_orth;
X = samples;




