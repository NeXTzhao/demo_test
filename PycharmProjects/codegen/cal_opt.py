#
# import casadi as ca
#
# # 优化变量
# x = ca.MX.sym("x")
#
# # 目标函数 f = (x - 1)^2
# f = (x - 1)**2
#
# # 约束 g = x >= 0
# g = x
#
# # 创建目标和约束函数
# nlp = ca.Function("nlp_func", [x], [f, g])
# nlp.generate("nlp_func", {
#     "with_header": True,
#     "with_export": True,
#     "cpp": True
# })
#
# # 生成梯度、Jacobian、Hessian
# grad_f = ca.Function("grad_f", [x], [ca.gradient(f, x)])
# grad_f.generate("grad_f", {
#     "with_header": True,
#     "cpp": True
#
# })
#
# jac_g = ca.Function("jac_g", [x], [ca.jacobian(g, x)])
# jac_g.generate("jac_g", {
#     "with_header": True,
#     "cpp": True
#
# })
#
# lam_g = ca.MX.sym("lam_g", g.shape[0])
# L = f + ca.dot(lam_g, g)
# hess_lag = ca.Function("hess_lag", [x, lam_g], [ca.hessian(L, x)[0]])
# hess_lag.generate("hess_lag", {
#     "with_header": True,
#     "with_mem": True,
#     "cpp": True
#
# })

import casadi as ca

# 定义输入
x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 1)

# 定义函数
y = ca.vertcat(ca.sin(x[0] + u[0]),
               x[1] * u[0])

# 构造函数对象
f = ca.Function("f", [x, u], [y])

# 导出原函数
f.generate("f", {
    "with_header": True,
    "cpp": True

})

# 导出 Jacobian df/dx
Jx = ca.Function("Jx", [x, u], [ca.jacobian(y, x)])
Jx.generate("Jx",  {
    "with_header": True,
    "cpp": True

})

# 导出 Hessian d²f₀/dx²（第一个输出的 Hessian）
hess = ca.Function("H", [x, u], [ca.hessian(y[0], x)[0]])
hess.generate("H",  {
    "with_header": True,
    "cpp": True

})
