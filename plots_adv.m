clear; clc;

V1 = load('/Users/admin/Desktop/DDPG/mg_rtds/adv/Voltage_d_4.csv');
V2 = load('/Users/admin/Desktop/DDPG/mg_rtds/adv/Voltage_d_5.csv');
V1(1) = [];
V2(1) = [];
X = linspace(0,5,1000).';

tiledlayout(2,1)

xl = 0;
xu = 0.4;
yl = 0.45;
yu = 0.51;
fs = 23;

% Top plot
nexttile
plot(X,V1,'LineWidth',1.5)
xlim([xl xu])
ylim([yl yu])
yline(0.48,'-r')
xline(X(fs),':m','LineWidth',1.5)
yline(0.4944,'-.k','LineWidth',1.5)
yline(0.4656,'-.k','LineWidth',1.5)
xlabel('Time (s)','FontSize',13)
ylabel('Voltage','FontSize',13)

% Bottom plot
nexttile
plot(X,V2,'LineWidth',1.5)
xlim([xl xu])
ylim([yl yu])
yline(0.48,'-r')
xline(X(fs),':m','LineWidth',1.5)
yline(0.4944,'-.k','LineWidth',1.5)
yline(0.4656,'-.k','LineWidth',1.5)
xlabel('Time (s)','FontSize',13)
ylabel('Voltage','FontSize',13)
