def _check_direction(mode, c_price, d_price):
    if mode == 1: return d_price < c_price # Bullish: D is low, C is high
    if mode == -1: return d_price > c_price # Bearish: D is high, C is low
    return False

def is_bat(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.382 and xab <= 0.5
    _abc = abc >= 0.382 and abc <= 0.886
    _bcd = bcd >= 1.618 and bcd <= 2.618
    _xad = xad <= 0.618
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_gartley(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.5 and xab <= 0.618
    _abc = abc >= 0.382 and abc <= 0.886
    _bcd = bcd >= 1.13 and bcd <= 2.618
    _xad = xad >= 0.75 and xad <= 0.875
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_abcd(xab, abc, bcd, xad, mode, c_price, d_price):
    _abc = abc >= 0.382 and abc <= 0.886
    _bcd = bcd >= 1.13 and bcd <= 2.618
    return _abc and _bcd and _check_direction(mode, c_price, d_price)

def is_anti_bat(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.500 and xab <= 0.886
    _abc = abc >= 1.000 and abc <= 2.618
    _bcd = bcd >= 1.618 and bcd <= 2.618
    _xad = xad >= 0.886 and xad <= 1.000
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_alt_bat(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab <= 0.382
    _abc = abc >= 0.382 and abc <= 0.886
    _bcd = bcd >= 2.0 and bcd <= 3.618
    _xad = xad <= 1.13
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_butterfly(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab <= 0.786
    _abc = abc >= 0.382 and abc <= 0.886
    _bcd = bcd >= 1.618 and bcd <= 2.618
    _xad = xad >= 1.27 and xad <= 1.618
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_anti_butterfly(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.236 and xab <= 0.886
    _abc = abc >= 1.130 and abc <= 2.618
    _bcd = bcd >= 1.000 and bcd <= 1.382
    _xad = xad >= 0.500 and xad <= 0.886
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_anti_gartley(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.500 and xab <= 0.886
    _abc = abc >= 1.000 and abc <= 2.618
    _bcd = bcd >= 1.500 and bcd <= 5.000
    _xad = xad >= 1.000 and xad <= 5.000
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_crab(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.500 and xab <= 0.875
    _abc = abc >= 0.382 and abc <= 0.886
    _bcd = bcd >= 2.000 and bcd <= 5.000
    _xad = xad >= 1.382 and xad <= 5.000
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_anti_crab(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.250 and xab <= 0.500
    _abc = abc >= 1.130 and abc <= 2.618
    _bcd = bcd >= 1.618 and bcd <= 2.618
    _xad = xad >= 0.500 and xad <= 0.750
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_shark(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.500 and xab <= 0.875
    _abc = abc >= 1.130 and abc <= 1.618
    _bcd = bcd >= 1.270 and bcd <= 2.240
    _xad = xad >= 0.886 and xad <= 1.130
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_anti_shark(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.382 and xab <= 0.875
    _abc = abc >= 0.500 and abc <= 1.000
    _bcd = bcd >= 1.250 and bcd <= 2.618
    _xad = xad >= 0.500 and xad <= 1.250
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_5o(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 1.13 and xab <= 1.618
    _abc = abc >= 1.618 and abc <= 2.24
    _bcd = bcd >= 0.5 and bcd <= 0.625
    _xad = xad >= 0.0 and xad <= 0.236
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_wolf(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 1.27 and xab <= 1.618
    _abc = abc >= 0 and abc <= 5
    _bcd = bcd >= 1.27 and bcd <= 1.618
    _xad = xad >= 0.0 and xad <= 5
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_hns(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 2.0 and xab <= 10
    _abc = abc >= 0.90 and abc <= 1.1
    _bcd = bcd >= 0.236 and bcd <= 0.88
    _xad = xad >= 0.90 and xad <= 1.1
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_con_tria(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 0.382 and xab <= 0.618
    _abc = abc >= 0.382 and abc <= 0.618
    _bcd = bcd >= 0.382 and bcd <= 0.618
    _xad = xad >= 0.236 and xad <= 0.764
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

def is_exp_tria(xab, abc, bcd, xad, mode, c_price, d_price):
    _xab = xab >= 1.236 and xab <= 1.618
    _abc = abc >= 1.000 and abc <= 1.618
    _bcd = bcd >= 1.236 and bcd <= 2.000
    _xad = xad >= 2.000 and xad <= 2.236
    return _xab and _abc and _bcd and _xad and _check_direction(mode, c_price, d_price)

# List of all pattern checkers for easy iteration
ALL_PATTERNS = [
    {'name': 'Bat', 'func': is_bat},
    {'name': 'Anti Bat', 'func': is_anti_bat},
    {'name': 'Alt Bat', 'func': is_alt_bat},
    {'name': 'Butterfly', 'func': is_butterfly},
    {'name': 'Anti Butterfly', 'func': is_anti_butterfly},
    {'name': 'ABCD', 'func': is_abcd},
    {'name': 'Gartley', 'func': is_gartley},
    {'name': 'Anti Gartley', 'func': is_anti_gartley},
    {'name': 'Crab', 'func': is_crab},
    {'name': 'Anti Crab', 'func': is_anti_crab},
    {'name': 'Shark', 'func': is_shark},
    {'name': 'Anti Shark', 'func': is_anti_shark},
    {'name': '5-O', 'func': is_5o},
    {'name': 'Wolf Wave', 'func': is_wolf},
    {'name': 'H&S', 'func': is_hns},
    {'name': 'Cont. Triangle', 'func': is_con_tria},
    {'name': 'Exp. Triangle', 'func': is_exp_tria},
] 