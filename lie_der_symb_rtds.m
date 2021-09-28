function [lie_der] = lie_der_symb_rtds(h,f)
%% Lie-derivative
u1 = sym('u1');
u2 = sym('u2');
f(5) = f(5) + u1;
f(6) = f(6) + u2;
var_suffix_global = {1:10};
var_suffix = var_suffix_global{1};

for i = 1:numel(var_suffix)
    temp_x = sym(['x' num2str(var_suffix(i))]);
    temp_x_t  = str2sym(['x' num2str(var_suffix(i)) '(t)']);
    temp_dx_dt = str2sym(['diff(x' num2str(var_suffix(i)) '(t), t)']);

    x(i) = temp_x;
    x_t(i) = temp_x_t;
    dx_dt(i) = temp_dx_dt;
end

h_t = subs(h, x, x_t);
dh_dt = diff(h_t);

%% substitute differentials
lie_der = subs(dh_dt, dx_dt, f.');
lie_der = subs(lie_der, x_t, x);
end
