function d = computeEmpiricalMeans(d, Pot, labels, type)

% updates empirical means
% Pot is the potential vector (unary) or matrix (pairwise) for an example
% labels are the GT labels (a scalar for unary, a 2D vector for pairwise)
% for pairwise, labels(1) is the row, labels(2) is the column
% labels should NOT be 0, they need to be indices into Pot!
% type ... 1 (unary), or 2 (pairwise)
% the function also supports multiple examples:
%    for unary, the rows are different examples, the columns are potentials for different classes
%    for pairwise, it's a 3D matrix, where the third dimensions are the examples
%    for pairwise, labels should be a nx2 matrix, where n is the number of examples

switch type
   case 1
      if length(labels)==1
         d = d + Pot(labels);
      else
         ind = sub2ind(size(Pot), [1:size(Pot, 1)]', labels);
         d = d + sum(Pot(ind));
      end;
   case 2
      if length(labels)==2
         d = d + Pot(labels(1), labels(2));
      else
         n = size(labels, 1);
         ind = (0:n-1)' * size(Pot, 1) * size(Pot, 2) + (labels(:, 2) - 1) * size(Pot, 1) + labels(:, 1);
         d = d + sum(Pot(ind));
      end;
   otherwise
      error('currently not supported');
end;


