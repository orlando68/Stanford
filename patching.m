v = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 1 1 1];
f = [1 3 2 4 5];
patch('Faces',f,'Vertices',v,'FaceColor','red')

patch([0,3,3,0],[0,0,3,3],[1,1,1,1],'blue')
% a= [0,3,3,0;0,0,3,3;2,2,2,2]
% patch(a,'red')