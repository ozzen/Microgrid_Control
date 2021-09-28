function [s] = rtds_ode(a,b,c,d,e,f,g,h,i,j,u1,u2)

i_d = [a];
i_q = [b];
i_od = [c];
i_oq = [d];
v_od = [e];
v_oq = [f];
i_ld = [g];
i_lq = [h];
m_d = [i];
m_q = [j];

t = 0:0.0032:0.0064;

[t,x] = ode45(@(t,x)PV(t,x,u1,u2),t,[i_d;i_q;i_od;i_oq;v_od;v_oq;i_ld;i_lq;m_d;m_q]);

z1 = {x(2,1)};
z2 = {x(2,2)};
z3 = {x(2,3)};
z4 = {x(2,4)};
z5 = {x(2,5)};
z6 = {x(2,6)};
z7 = {x(2,7)};
z8 = {x(2,8)};
z9 = {x(2,9)};
z10 = {x(2,10)};

s(1,1) =  z1;
s(1,2) =  z2;
s(1,3) =  z3;
s(1,4) =  z4;
s(1,5) =  z5;
s(1,6) =  z6;
s(1,7) =  z7;
s(1,8) =  z8;
s(1,9) =  z9;
s(1,10) =  z10;
