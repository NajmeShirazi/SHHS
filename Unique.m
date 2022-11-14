clc
clear all

%data = readtable('C:\Users\Najmeh\Desktop\New\Patients- Stroke- Heart Vessels- Heart Failure');
%A = table2cell(data);
%M = cell2mat(A);
M = xlsread('C:\Users\Najmeh\Desktop\New\Patients-Stroke-HeartVessels-HeartFailure1');

[~,idx] = unique(M(:,1));
H = M(idx,:);

nsrrid = H(:,1);
age_category_s1_65 = H(:,2);
prev_hrtvsl = H(:,3);
prev_chf = H(:,4);
prev_stk = H(:,5);
gender = H(:,6);
race = H(:,7);
bmi_s1 = H(:,8);
Chol = H(:,9);
HDL = H(:,10);
Trig = H(:,11);
HTNDerv_s1 = H(:,12);
ParRptDiab = H(:,13);
SA15 = H(:,14);
% Names = {'nsrrid'; 'age_category_s1_65'; 'prev_hrtvsl'; 'prev_chf'; 'prev_stk'; 'gender'; 'race'; 'bmi_s1'; 'Chol'; 'HDL'; 'Trig'; 'HTNDerv_s1'; 'ParRptDiab'; 'SA15'};
T = table(nsrrid,age_category_s1_65,prev_hrtvsl,prev_chf,prev_stk,gender,race,bmi_s1,Chol,HDL,Trig,HTNDerv_s1,ParRptDiab,SA15);
filename = 'Patients-Str-HrtVsl-HrtFlr.xlsx';
writetable(T, filename, 'Sheet', 'AllPatients')
winopen(filename);
