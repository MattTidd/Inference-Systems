% *************************************************************************
% FILE NAME        : csv2dat.m
% AUTHOR           : Matthew Tidd
% DATE CREATED     : 30 Oct 2024
% DATE MODIFIED    : 30 Oct 2024
% *************************************************************************
% Preamble:
% *************************************************************************

clc ; clear ; close all ; format short;

% *************************************************************************
% Main:
% *************************************************************************

% need to load the csv files and save them in a .dat format so that the
% anfis app can load the data in:

input_data = readmatrix('inputs.csv');
output_data = readmatrix('outputs.csv');

data = [input_data output_data];

writematrix(data, 'train_data.dat', 'Delimiter', ' ')
writematrix(input_data, 'inputs.dat', 'Delimiter', ' ');
writematrix(output_data, 'outputs.dat', 'Delimiter', ' ')

% check that it worked:

fid = fopen('inputs.dat');
dat_data = fread(fid, '*char').';
fclose(fid);
disp(dat_data)