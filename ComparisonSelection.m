clc
clear all

% data = readtable('D:\My Project- Dr.Safavi\Complete\SHHS1');
% A = table2cell(data);
% M = cell2mat(A);

M = xlsread('C:\Users\Najmeh\Desktop\New\Result2\NormalNotAbs1.xlsx');
M1 = xlsread('C:\Users\Najmeh\Desktop\New\Result2\Normal-SF1.xlsx');

[r,c] = size(M);
for i=1:r
idx(i) = M(i);
[id,~] = find(M1(:,1)==idx);
H = M1(id,:);
end


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
filename = 'NormalNotAbs.xlsx';
writetable(T, filename, 'Sheet', 'NormalNotAbs')
winopen(filename);
