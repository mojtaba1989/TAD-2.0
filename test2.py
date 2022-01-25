import numpy as np


def decimal2binary_8bit(input):
    output = np.zeros((8,), dtype=int)
    if input < 0:
        flag = 1
        input = - input
    else:
        flag = 0
    for i in range(8):
        output[i] = input % 2
        input = int(input / 2)
    if flag:
        output = twosComplement_8bits(output)
    return output


def binary2decimal_8bit(input):
    if input[7]:
        twosComplement_8bits(input)
        sign = 1
    else:
        sign = 0
    for i in range(8):
        try:
            output += input[i] * 2 ** i
        except:
            output = input[i] * 2 ** i
    if sign:
        output = -output
        twosComplement_8bits(input)
    return output


def decimal2binary_bfloat16(input):
    output = np.zeros((16,), dtype=int)
    if input < 0:
        flag = 1
        input = - input
    else:
        flag = 0
    input_int = int(input)
    input_frac = input - input_int
    for i in range(7, 16):
        output[i] = input_int % 2
        input_int = int(input_int / 2)

    for i in range(7):
        input_frac = input_frac * 2
        output[6 - i] = int(input_frac)
        input_frac -= output[6 - i]
    if flag:
        twosComplement_16bits(output)
    return output


def binary2decimal_bfloat16(input):
    if input[15]:
        sign = 1
        input = twosComplement_16bits(input)
    else:
        sign = 0
    for i in range(16):
        try:
            output += input[i] * 2 ** (i - 7)
        except:
            output = input[i] * 2 ** (i - 7)
    if sign:
        output = - output
        twosComplement_16bits(input)
    return output


def xor(A, B):
    output = ((not A) and B) or (A and (not B))
    return output


def one_bit_full_adder(A, B, Cin):
    s = xor(A, xor(B, Cin))
    Cout = (A and B) or (A and Cin) or (B and Cin)
    return s, Cout


def twosComplement_8bits(A):
    Cin = 1
    for i in range(8):
        A[i] = xor(A[i], 1)
        A[i], Cin = one_bit_full_adder(A[i], 0, Cin)
    return A


def twosComplement_16bits(A):
    Cin = 1
    for i in range(16):
        A[i] = xor(A[i], 1)
        A[i], Cin = one_bit_full_adder(A[i], 0, Cin)
    return A


def full_adder_subtractor_8bits(A, B, subtract=0):  # Add: subtract = 0; Subtract: subtract = 1
    output = np.zeros((8,), dtype=int)
    if not subtract:
        Cin = 0
        for i in range(8):
            output[i], Cin = one_bit_full_adder(A[i], B[i], Cin)
    else:
        B = twosComplement_8bits(B)
        output = full_adder_subtractor_8bits(A, B)
        B = twosComplement_8bits(B)
    return output


def full_adder_subtractor_16bits(A, B, subtract=0):  # Add: subtract = 0; Subtract: subtract = 1
    output = np.zeros((16,), dtype=int)
    if not subtract:
        Cin = 0
        for i in range(16):
            output[i], Cin = one_bit_full_adder(A[i], B[i], Cin)
    else:
        B = twosComplement_16bits(B)
        output = full_adder_subtractor_16bits(A, B)
        B = twosComplement_16bits(B)
    return output


def shiftRegister(A, data, length, direction=0):  # direction( 0 -> left; 1-> right)
    if not direction:
        for i in reversed(range(1, length)):
            A[i] = A[i - 1]
        A[0] = data
    else:
        for i in range(length-1):
            A[i] = A[i + 1]
        A[length - 1] = data
    return A


def multiplier(A, B):
    output = np.zeros((8,), dtype=int)
    temp = np.zeros((8,), dtype=int)
    sign = xor(A[7], B[7])
    if A[7]: A = twosComplement_8bits(A)
    if B[7]: B = twosComplement_8bits(B)
    for i in range(8):
        for j in range(8):
            temp[j] = A[j] and B[i]
        A = shiftRegister(A, 0, 8)
        output = full_adder_subtractor_8bits(output, temp)
    if sign: twosComplement_8bits(output)
    return output


def comparator(A, B, length=16):
    out = 1
    for i in range(length - 1, -1, -1):
        if xor(A[i], B[i]):
            out = A[i]
            break
    return out

def division(dividend, divisor):
    sign = xor(dividend[7], divisor[7])
    quotient = decimal2binary_bfloat16(0)
    dividend_16bit = decimal2binary_bfloat16(binary2decimal_8bit(dividend))
    divisor_16bit = decimal2binary_bfloat16(binary2decimal_8bit(divisor))
    if dividend[7]:
        twosComplement_16bits(dividend_16bit)
    if divisor[7]:
        twosComplement_16bits(divisor_16bit)
    flag = 1
    for i in range(8):
        if (not divisor_16bit[14]) and flag: shiftRegister(divisor_16bit, 0, 16, 0)
        else: shiftRegister(dividend_16bit, 0, 16, 1); flag=0
    for i in range(15):
        shiftRegister(quotient, comparator(dividend_16bit, divisor_16bit, 16), 16, 0)
        if quotient[0]:
            dividend_16bit = full_adder_subtractor_16bits(dividend_16bit, divisor_16bit, 1)
        shiftRegister(divisor_16bit, 0, 16, 1)
    shiftRegister(quotient, 0, 16, 0)
    if sign: twosComplement_16bits(quotient)
    return quotient

if __name__ == "__main__":
    intIn = 7
    pOut = decimal2binary_8bit(intIn)
    print(intIn, '=', pOut)

    # binary2decimal_8bit
    intOut = binary2decimal_8bit(pOut)
    print(pOut, '=', intOut)

    # decimal2binary_bfloat16
    doubleIn = -27.675777
    pOut = decimal2binary_bfloat16(doubleIn)
    print(doubleIn, '=', pOut)

    # binary2decimal_bfloat16
    print(pOut, end = '')
    doubleOut = binary2decimal_bfloat16(pOut)
    print( '=', doubleOut)

    # xor
    intOut = xor(1, 0)
    print('xor(1, 0)=', intOut)

    # one_bit_full_adder
    pOut, Cout = one_bit_full_adder(0, 1, 0)
    print('0+1+0-->', 'S =', pOut, 'Cout=', Cout)

    # twosComplement

    pInt1 = decimal2binary_8bit(13)
    print(pInt1, end =" ")
    print(" ---- two's complement---> ", end =" ")
    twosComplement_8bits(pInt1)
    print(pInt1)

    pInt1 = decimal2binary_bfloat16(13.5)
    print(pInt1, end =" ")
    print(" ---- two's complement---> ", end =" ")
    twosComplement_16bits(pInt1)
    print(pInt1)

    # shiftRegister
    print(pInt1, end =" ")
    print(" --right shift bye 1 --> ", end =" ")
    shiftRegister(pInt1, 0, 16, 1)
    print(pInt1, end =" ")
    print(" --left shift bye 2 --> ", end =" ")
    shiftRegister(pInt1, 0, 16, 0)
    shiftRegister(pInt1, 0, 16, 0)
    print(pInt1)

    # full_adder_8bits
    pInt1 = decimal2binary_8bit(13)
    pInt2 = decimal2binary_8bit(10)
    pOut = full_adder_subtractor_8bits(pInt1, pInt2, 0)
    intOut = binary2decimal_8bit(pOut)
    print(pInt1, "+", pInt2, '=', pOut, '=', intOut)

    pOut = full_adder_subtractor_8bits(pInt1, pInt2, 1)
    intOut = binary2decimal_8bit(pOut)
    print(pInt1, "-", pInt2, '=', pOut, '=', intOut)

    pOut = full_adder_subtractor_8bits(pInt2, pInt1, 1)
    intOut = binary2decimal_8bit(pOut)
    print(pInt2, "-", pInt1, '=', pOut, '=', intOut)

    # multiplier
    pInt1 = decimal2binary_8bit(7)
    pInt2 = decimal2binary_8bit(-5)
    pOut = multiplier(pInt2, pInt1)
    intOut = binary2decimal_8bit(pOut)
    print(pInt2, "*", pInt1, '=', pOut, '=', intOut)

    # comparator
    pInt1 = decimal2binary_8bit(8)
    pInt2 = decimal2binary_8bit(5)
    intOut=comparator(pInt1, pInt2, 8)
    print(pInt2, ">=", pInt1, '?', intOut)

    # division
    pInt1 = decimal2binary_8bit(8)
    pInt2 = decimal2binary_8bit(1)
    pOut = division(pInt1, pInt2)
    intOut = binary2decimal_bfloat16(pOut)
    print(pInt1, "/", pInt2, '=', pOut, '=', intOut)






