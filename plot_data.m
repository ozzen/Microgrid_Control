clear; clc;
Voltage = load('/Users/admin/Desktop/DDPG/mg_rtds/adv/Voltage_d_10.csv');
Voltage(1) = [];
Voltage=Voltage;
% Plotting
X = linspace(0,10,500).';
fs = 170;
plot(X(1:500),Voltage(1:500),'LineWidth',1.5)
yline(0.48,'-r','LineWidth',2.5)
yline(0.4944,'-.k','LineWidth',2.5)
yline(0.4656,'-.k','LineWidth',2.5)
xline(X(fs),':m','LineWidth',2)
% xline(X(fs+12),':m','LineWidth',2.5)
ylim([0.459 0.485])
% xlim([0 1])
box on
ax = gca;
ax.LineWidth = 1;
xlabel('Time (s)','FontSize',22)
ylabel('Voltage (kV)','FontSize',22)
set(gca,'FontSize',25)
% title('NC with AM-I')
% title('NC without AM')
% title('NC without BC')
% title('NC-RTDS with adversarial inputs');

%% Prelims
batch = 1;
episode = 500;
ref = 0.48;
dev = 0.005;
dt = 0.005;

V = reshape(Voltage,episode,batch);
V_error = abs(V - ref);
conv(1:episode,1:batch) = 0;

% %% Average Convergence Time
% for i = 1:batch
%     for j = 1:episode
%         if V(j,i) < ref+dev && V(j,i) > ref-dev
%             conv(j,i) = j;
%         else
%             conv(:,i) = 0;
%         end
%     end
% end
% 
% for i = 1:batch
%     T(1,i) = find(conv(:,i)~=0, 1, 'first');
% end
% 
% avg_conv_time = mean(T) * dt;
% %% Mean Absolute Error
% for i = 1:batch
%     tau = T(1,i);
%     error(1,i) = sum(V_error(tau:episode,i)) / (episode - tau);
% end
% 
% mean_abs_error = mean(error);
% %% Maximum Deviation
% for i = 1:batch
%     tau = T(1,i);
%     max_dev_batch(1,i) = max(V(tau:episode,i));
% end
% 
% [max_dev,idx] = max(max_dev_batch);
% max_dev = abs(max_dev - ref);
