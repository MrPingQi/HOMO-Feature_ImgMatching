function [points,solution_t] = XY_Transform(points,solutions,flag)

solution_t = solutions;
solution_t(1:end,1) = {[1,0,0;0,1,0;0,0,1]};
solution_t(1:end,2) = {'affine'};

if isempty(points)
    return
end
points = [points(:,1:2),ones(size(points,1),1)];

k = 1;
for i = 1:size(solutions,1)
    tform = solutions{i,2}; 
    if strcmpi(tform,'nonreflectivesimilarity') || strcmpi(tform,'similarity') || strcmpi(tform,'affine')
        solution_t{k,1} = solution_t{k,1}*solutions{i,1};
    elseif strcmpi(tform,'projective')
        if i~=1
            k = k+1;
        end
        solution_t(k,:) = solutions(i,:);
        if i~=size(solutions,1)
            k = k+1;
        end
    end
end
solution_t = solution_t(1:k,:);

if flag==1
    for i = 1:size(solution_t,1)
        tform = solution_t{i,2}; 
        if strcmpi(tform,'affine')
            points = points*solution_t{i,1};
        elseif strcmpi(tform,'projective')
            M = solution_t{i,1};
            W = points(:,1)*M(1,3)+points(:,2)*M(2,3)+M(3,3);
            points(:,1:2) = ...
                [(points(:,1)*M(1,1)+points(:,2)*M(2,1)+M(3,1)) ./ W,...
                 (points(:,1)*M(1,2)+points(:,2)*M(2,2)+M(3,2)) ./ W];
        end
    end
    
elseif flag==-1
    for i = size(solution_t,1):-1:1
        tform = solution_t(i,2); 
        if strcmpi(tform,'affine')
            points = points/solution_t{i,1};
        elseif strcmpi(tform,'projective')
            M = inv(solution_t{i,1});
            W = points(:,1)*M(1,3)+points(:,2)*M(2,3)+M(3,3);
            points(:,1:2) = ...
                [(points(:,1)*M(1,1)+points(:,2)*M(2,1)+M(3,1)) ./ W,...
                 (points(:,1)*M(1,2)+points(:,2)*M(2,2)+M(3,2)) ./ W];
        end
    end
end

points = points(:,1:2);