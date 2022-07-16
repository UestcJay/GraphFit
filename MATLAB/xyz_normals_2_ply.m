
clear all
close all
clc
data_path = '../log/outputs/';
normals_path = '../log/outputs/';
file_name = '0000000000';
pc_filename = [data_path, file_name, '.xyz'];
normals_filename = [normals_path, file_name, '.normals'];
output_ply_filename = [normals_path, file_name, '.ply'];
points= dlmread(pc_filename);
normals = dlmread(normals_filename);

normals = sign(sum(points.*normals,2)).*normals; %assume normal vectors point from the origin outwards and reorient the normals
pcCloud_obj = pointCloud(points, 'Normal', normals);
pcwrite(pcCloud_obj,output_ply_filename)