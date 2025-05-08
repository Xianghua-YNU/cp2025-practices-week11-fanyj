"""
均匀薄片引力计算 - 学生模板

本模板用于计算方形薄片在垂直方向上的引力分布，学生需要完成以下部分：
1. 实现高斯-勒让德积分方法
2. 计算不同高度处的引力值
3. 绘制引力随高度变化的曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad  # 引入scipy的二重积分函数

# 物理常数
G = 6.67430e-11  # 万有引力常数 (单位: m^3 kg^-1 s^-2)


def calculate_sigma(length, mass):
    """
    计算薄片的面密度

    参数:
        length: 薄片边长 (m)
        mass: 薄片总质量 (kg)

    返回:
        面密度 (kg/m^2)
    """
    return mass / (length ** 2)


def integrand(x, y, z):
    """
    被积函数，计算引力积分核

    参数:
        x, y: 薄片上点的坐标 (m)
        z: 测试点高度 (m)

    返回:
        积分核函数值
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return z / (r ** 3)


def gauss_legendre_integral(length, z, n_points=100):
    """
    使用高斯-勒让德求积法计算二重积分

    参数:
        length: 薄片边长 (m)
        z: 测试点高度 (m)
        n_points: 积分点数 (默认100)

    返回:
        积分结果值

    提示:
        1. 使用np.polynomial.legendre.leggauss获取高斯点和权重
        2. 将积分区间从[-1,1]映射到[-L/2,L/2]
        3. 实现双重循环计算二重积分
    """
    x_points, x_weights = np.polynomial.legendre.leggauss(n_points)
    y_points, y_weights = np.polynomial.legendre.leggauss(n_points)

    x_mapped = 0.5 * length * x_points
    y_mapped = 0.5 * length * y_points

    integral = 0
    for i in range(n_points):
        for j in range(n_points):
            integral += x_weights[i] * y_weights[j] * integrand(x_mapped[i], y_mapped[j], z)

    return (0.25 * length ** 2) * integral


def calculate_force(length, mass, z, method='gauss'):
    """
    计算给定高度处的引力

    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z: 测试点高度 (m)
        method: 积分方法 ('gauss'或'scipy')

    返回:
        引力值 (N)
    """
    sigma = calculate_sigma(length, mass)
    if method == 'gauss':
        integral_result = gauss_legendre_integral(length, z)
    elif method =='scipy':
        # 使用scipy进行二重积分计算
        integral_result, _ = dblquad(lambda y, x: integrand(x, y, z),
                                     -length / 2, length / 2,
                                     lambda x: -length / 2, lambda x: length / 2)
    else:
        raise ValueError("仅支持 'gauss' 或'scipy' 积分方法。")
    return G * sigma * integral_result


def plot_force_vs_height(length, mass, z_min=0.1, z_max=10, n_points=100):
    """
    绘制引力随高度变化的曲线

    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z_min: 最小高度 (m)
        z_max: 最大高度 (m)
        n_points: 采样点数
    """
    z_array = np.linspace(z_min, z_max, n_points)
    force_array = np.array([calculate_force(length, mass, z) for z in z_array])

    plt.plot(z_array, force_array, label='计算得到的引力')

    # 添加理论极限线
    plt.axhline(y=G * mass / (z_max ** 2), color='r', linestyle='--', label='理论极限')

    plt.title('引力随高度变化曲线')
    plt.xlabel('高度 (m)')
    plt.ylabel('引力 (N)')
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例使用
if __name__ == '__main__':
    # 参数设置 (边长10m，质量1e4kg)
    length = 10
    mass = 1e4

    # 计算并绘制引力曲线
    plot_force_vs_height(length, mass)

    # 打印几个关键点的引力值
    for z in [0.1, 1, 5, 10]:
        F = calculate_force(length, mass, z)
        print(f"高度 z = {z:.1f}m 处的引力 F_z = {F:.3e} N")
