function [index] = bound(percent, data)
%BOUND	Percentile bound on a histogram
%	[INDEX] = bound(Percentile, X) finds the index for the value
%	of the histogram, X, which bounds Percentile events. for matrix X
%	BOUND finds said index of each column and returns a vector 
%	of indices.
%
%	See also: TMSSTAT, POSSTAT, RATSTAT

%	AJHansen 5 November 1996
%	modified by Todd Walter 11 June 1997

[m n] = size(data);
index = zeros(1,n);
total = sum(data);

for idx = 1:n
    include = 0;
    start = floor(m/2);
    bnd_cnt = ceil(percent*total);
    curr = data(start,idx) + data(start+1,idx);
    while (curr < bnd_cnt) & (include <= start)
        include = include + 1;
        curr = curr + data(start-include,idx) + data(start+1+include,idx);
    end
    index(idx) = start + include + 1;
end
