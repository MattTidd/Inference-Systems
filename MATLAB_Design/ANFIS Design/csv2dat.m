% *************************************************************************
% FILE NAME        : csv2dat.m
% AUTHOR           : Matthew Tidd
% DATE CREATED     : 11 Dec 2024
% DATE MODIFIED    : 12 Dec 2024
% *************************************************************************
% Preamble:
% *************************************************************************

clc ; clear ; close all ; format short;

% *************************************************************************
% Main:
% *************************************************************************

% get data path:
subfolder_name = 'Data';
folder_path = fullfile(pwd, subfolder_name);

% import data:
train_input_data = readmatrix(fullfile(folder_path, "\CSVs\train_input.csv"));
train_output_data = readmatrix(fullfile(folder_path, "\CSVs\train_output.csv"));
test_input_data = readmatrix(fullfile(folder_path, "\CSVs\test_input.csv"));
test_output_data = readmatrix(fullfile(folder_path, "\CSVs\test_output.csv"));

% convert to .dat format: 
writematrix(train_input_data, fullfile(folder_path,"\DATs\train_input.dat"), 'Delimiter', ' ');
writematrix(train_output_data, fullfile(folder_path,"\DATs\train_output.dat"), 'Delimiter', ' ');
writematrix(test_input_data, fullfile(folder_path,"\DATs\test_input.dat"), 'Delimiter', ' ');
writematrix(test_output_data, fullfile(folder_path,"\DATs\test_output.dat"), 'Delimiter', ' ');