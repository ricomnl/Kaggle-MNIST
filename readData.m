function [X y] = readData(filename);
  
X = csvread(filename);
X = X(2:end, :);
y = X(:, 1);
X = X(:, 2:end);

end