clear; clc;

n = 1;
low = 0;
init_d = 0.2;
init_q = 0.000001;
high_i_d = 1.0;
high_v_d = 1.0;
high_q = 0.00001;
ref = 0.48;
fsc = 0.0144;
w = 60;
I_ld = 0.6;
I_lq = 0;
K_p =  0.5;

i_d = init_state(low, init_d, n);
i_q = init_state(low, init_q, n);
i_od = init_state(low, init_d, n);
i_oq = init_state(low, init_q, n);
v_od = init_state(ref-fsc, ref+fsc, n);
v_oq = init_state(low, init_q, n);
i_ld = init_state(low, init_d, n);
i_lq = init_state(low, init_q, n);
m_d = init_state(low, init_d, n);
m_q = init_state(low, init_q, n);
s = [i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q];

u1 = 0.087*m_d - v_od;
u2 = 0.087*m_q - v_oq;

for i = 1:999
    t = 0:0.005:0.01;
    [t,x] = ode45(@(t,x)PV(t,x,u1,u2),t,[i_d;i_q;i_od;i_oq;v_od;v_oq;i_ld;i_lq;m_d;m_q]);
    s(i+1,1:10) = x(2,:);
    s(i,11) = u1;
    s(i,12) = u2;
    
    s(i+1,1) = bound(s(i+1,1), low, high_i_d);
    s(i+1,2) = bound(s(i+1,2), low, high_q);
    s(i+1,3) = bound(s(i+1,3), low, high_i_d);
    s(i+1,4) = bound(s(i+1,4), low, high_q);
    s(i+1,5) = bound(s(i+1,5), low, high_v_d);
    s(i+1,6) = bound(s(i+1,6), low, high_q);
    s(i+1,7) = bound(s(i+1,7), low, high_i_d);
    s(i+1,8) = bound(s(i+1,8), low, high_q);
    s(i+1,9) = bound(s(i+1,9), low, high_i_d);
    s(i+1,10) = bound(s(i+1,10), low, high_q);
        
    u1 = 0.087*s(i+1,9) - s(i+1,5);
    u2 = 0.087*s(i+1,10) - s(i+1,6);        
    
    i_d = s(i+1,1);
    i_q = s(i+1,2);
    i_od = s(i+1,3);
    i_oq = s(i+1,4);
    v_od = s(i+1,5);
    v_oq = s(i+1,6);
    i_ld = s(i+1,7);
    i_lq = s(i+1,8);
    m_d = s(i+1,9);
    m_q = s(i+1,10);
end
plot(s(:,5))
