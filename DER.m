function f = PV(t,z,u1,u2)

R_f = 0.038;
C_f = 2500;
P_r = 0.5;
Q_r = 0.000001;
K_p = 0.5;
r_n = 0.01;

i_d = z(1);   % d-current
i_q = z(2);   % q-current
i_od = z(3);  % d-o/p of inverter
i_oq = z(4);  % q-o/p of inverter
v_od = z(5);  % d-i/p to voltage controller
v_oq = z(6);  % q-i/p to voltage controller
i_ld = z(7);  % d-i/p to current controller
i_lq = z(8);  % q-i/p to current controller
m_d = z(9);   % d-o/p to current controller
m_q = z(10);  % q-o/p to current controller

w = 60;
I_ld = 0.6;
I_lq = 0;
v_bd = r_n*(i_od-i_d); %0.48;
v_bq = r_n*(i_oq-i_q); %0.000001;

f = zeros(size(z));

f(1) = -P_r*i_d + w*i_q + v_bd;
f(2) = -Q_r*i_q - w*i_d + v_bq;
f(3) = -i_od + w*i_oq + v_od - v_bd;
f(4) = -i_oq - w*i_od + v_oq - v_bq;
f(5) = w*v_oq + (i_ld - i_od)/C_f + u1;
f(6) = -w*v_od + (i_lq - i_oq)/C_f + u2;
f(7) = -R_f*i_ld + w*i_lq + m_d - v_od;
f(8) = -R_f*i_lq - w*i_ld + m_q - v_oq;
f(9) = -w*i_lq + K_p*(I_ld - i_ld);
f(10) = -w*i_ld + K_p*(I_lq - i_lq);
