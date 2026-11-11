Q = 2**60 - 93
delta = 2**40
delta_inv = pow(delta, -1, Q)


def centered(x):
    x %= Q
    if x > Q / 2:
        return x - Q
    return x


for e in [1, -2, 3, 5, -7]:
    print(f"e = {e:3d},  delta^-1 * e mod Q = {centered(delta_inv * e):>20d}")
