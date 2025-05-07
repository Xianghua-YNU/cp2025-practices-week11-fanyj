import numpy as np
import matplotlib.pyplot as plt

# --- 物理和线圈参数 ---
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (T*m/A)
I = 1.0  # 电流 (A) - 假设为1A，实际计算中常数因子可以合并


def Helmholtz_coils(r_low, r_up, d):
    '''
    Calculate the Magnetic Field of Helmhotz coils
    input:
        r_low: radius of lower coil
        r_up : radius of upper coil
        d: distance between the two coils
    return:
        X,Y: 空间坐标
        By, Bz： y,z方向的磁场
    '''
    phi = np.linspace(0, 2 * np.pi, 20)  # 角度
    r = max(r_low, r_up)
    y = np.linspace(-2 * r, 2 * r, 25)
    z = np.linspace(-2 * d, 2 * d, 25)

    Y, Z, phi = np.meshgrid(y, z, phi)

    # Calculate the square root of the distance between the point and dl of the coils
    r1 = np.sqrt((r_low * np.cos(phi)) ** 2 + (Y - r_low * np.sin(phi)) ** 2 + (Z - d / 2) ** 2)  # 到第一个环的距离
    r2 = np.sqrt((r_up * np.cos(phi)) ** 2 + (Y - r_up * np.sin(phi)) ** 2 + (Z + d / 2) ** 2)  # 到第二个环的距离

    dby = r_low * (Z - d / 2) * np.sin(phi) / r1 ** 3 + r_up * (Z + d / 2) * np.sin(phi) / r2 ** 3  # 角度phi处上下两个电流元产生的y方向磁场
    dbz = r_low * (r_low - Y * np.sin(phi)) / r1 ** 3 + r_up * (r_up - Y * np.sin(phi)) / r2 ** 3  # 角度phi处上下两个电流元产生的z方向磁场

    # 考虑到兼容性，使用 np.trapz 替代 np.trapezoid
    By_unscaled = np.trapz(dby, x=phi, axis=-1)  # y方向的磁场，对整个电流环积分
    Bz_unscaled = np.trapz(dbz, x=phi, axis=-1)  # z方向的磁场，对整个电流环积分

    scaling_factor = (MU0 * I) / (4 * np.pi)
    By = scaling_factor * By_unscaled
    Bz = scaling_factor * Bz_unscaled

    return Y, Z, By, Bz


def plot_magnetic_field_streamplot(r_low, r_up, d):
    Y, Z, by, bz = Helmholtz_coils(r_low, r_up, d)

    bSY = np.arange(-0.45, 0.50, 0.05)  # 磁力线的起点的y坐标
    bSY, bSZ = np.meshgrid(bSY, 0)  # 磁力线的起点坐标
    points = np.vstack([bSY.ravel(), bSZ.ravel()]).T

    h1 = plt.streamplot(Y[:, :, 0], Z[:, :, 0], by, bz,
                        density=2, color='k', start_points=points)

    plt.xlabel('y / m')
    plt.ylabel('z / m')
    plt.title(f'Magnetic Field Lines of Helmholtz Coils (R1={r_low}, R2={r_up}, d={d})')
    plt.grid(True, linestyle='--', alpha=0.7)
    # 由于没有实际的 label，这里暂时不调用 legend
    # plt.legend()
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    coil_radius = 0.5  # 两个线圈的半径 (m)
    coil_distance = 0.8  # 两个线圈之间的距离 (m)
    plot_magnetic_field_streamplot(coil_radius, coil_radius, coil_distance)
    
