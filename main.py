import numpy as np
def nMathar(wv, P, T, H):
    """
    Calculate the index of refraction as given by Mathar (2008): http://arxiv.org/pdf/physics/0610256v2.pdf and Noel et al. (2023).
    Includes three Mathar approximation and automatically decides which to use depending on the wavelength
    ***Only valid for between 0.7 to 2.5 microns and 2.8 to 4.2 microns!

    Inputs:
        wv: wavelength in microns
        P:  Pressure in Pa
        T:  Temperature in Kelvin
        H:  relative humidity in % (i.e. between 0 and 100)
    Return:
        n:  index of refraction
    """
    n = np.ones(np.size(wv))  # output. default to 1.
    wvnum = 1.e4 / wv  # cm^-1     # convert to wavenumbers

    # if it was passed in as a float, we need to convert it into a array for code reuse
    if not isinstance(wvnum, np.ndarray):
        wvnum = np.array([wvnum])

    # for region 0 wv <= 1.36 µm. Technically only valid for 0.7 to 1.36 µm
    region0 = np.where(wv <= 1.36)
    if np.size(region0) > 0:
        
        # This approximation uses wavelengths instead of wavenumbers
        wavelen0 = 0.77 # µm 

        # do a 5th order expansion
        powers = np.arange(0, 6)
        # calculate expansion coefficients
        coeffs = get_coeff_mathar(powers, P, T, H, 0)

        # sum of the power series expansion
        for coeff, power in zip(coeffs, powers):
            n[region0] += coeff * ((1e4/wvnum[region0] - wavelen0) ** power)

    # polynomial expansion in wavenumber
    # calcualate index of refraction by splitting it up by region
    # for region 1. < 2.65 microns, technically only valid for 1.3 to 2.5 microns
    region1 = np.where((wv < 2.8)&(wv>1.36))
    if np.size(region1) > 0:
        wvnum0 = 1.e4 / 2.25  # cm^-1 #series expand around this wavenumber

        # do a 5th order expansion
        powers = np.arange(0, 6)
        # calculate expansion coefficients
        coeffs = get_coeff_mathar(powers, P, T, H, 1)

        # sum of the power series expansion
        for coeff, power in zip(coeffs, powers):
            n[region1] += coeff * ((wvnum[region1] - wvnum0) ** power)
    # next for everything greater than 2.65 microns. technically valid for 2.8 - 4.2 microns
    region2 = np.where(wv >= 2.8)
    if np.size(region2) > 0:
        wvnum0 = 1.e4 / 3.4  # cm^-1 #series expand around this wavenumber

        # do a 5th order expansion
        powers = np.arange(0, 6)
        # calculate expansion coefficients
        coeffs = get_coeff_mathar(powers, P, T, H, 2)

        # sum of the power series expansion
        for coeff, power in zip(coeffs, powers):
            n[region2] += coeff * ((wvnum[region2] - wvnum0) ** power)

    # return a int/float if that is what is passed in
    if isinstance(wv, (int, float)):
        n = n[0]

    return n


def get_coeff_mathar(i, P, T, H, wvrange=1):
    """
    Calculate the coefficients for the polynomial series expansion of index of refraction (Mathar (2008))
    Includes three Mathar approximation and automatically decides which to use depending on the wavelength
    ***Only valid for between 0.7 and 2.5 microns! and 2.8 through 4.2 microns!

    Inputs:
        i:  degree of expansion in wavenumber
        P:  Pressure in Pa
        T:  Temperature in Kelvin
        H:  relative humiditiy in % (i.e. between 0 and 100)
        wvrange (int): 0 = (0.7-1.3 µm), 1 = (1.3-2.5 µm), 2 = (2.8 - 4.2 µm)
    Return:
        coeff:  Coefficient [cm^-i]
    """

    # name all the constants in the model
    # series expansion in evironment parameters
    T0 = 273.15 + 17.5  # Kelvin
    P0 = 75000  # Pa
    H0 = 10  # %

    # delta terms for the expansion
    dT = 1. / T - 1. / T0
    dP = P - P0
    dH = H - H0

    # loads and loads of coefficients, see equation 7 in Mathar (2008)
    # use the power (i.e. i=[0..6]) to index the proper coefficient for that order
    if wvrange == 0:
        # 0.7 µm to 1.36 µm approximation uses different reference numbers
        T0 = 280.65 # Kelvin
        P0 = 66625 # Pa
        H0 = 50 # %
        dT = 1. / T - 1. / T0
        dP = P - P0
        dH = H - H0

        cref = np.array([1.85566259e-04, -4.68511206e-06, 9.19681919e-06, -1.44638085e-05, 1.52286899e-05, -7.42131053e-06])
        cT = np.array([5.33344343e-02, -1.24712782e-03, 2.33119745e-03, -2.32913516e-03, 1.75945139e-06, 1.51989359e-03])
        cTT = np.array([4.37191645e-06, -6.25121335e-08, 1.63938942e-07, -2.11103761e-07, -1.52898469e-08, 1.13124404e-07])
        cH = np.array([-5.29847992e-09, -3.13820651e-10, 4.69827651e-10, -3.50677283e-09, 9.63769669e-09, -9.13487764e-09])
        cHH = np.array([1.72638330e-13, 1.61933914e-12, -5.64003179e-12, -2.62670875e-12, 1.21144700e-11, 4.26582641e-12])
        cP = np.array([2.78974970e-09, -7.00536198e-11, 1.37565581e-10, -2.14757969e-10, 2.22197137e-10, -1.04766954e-10])
        cPP = np.array([2.26729683e-17, 7.56136386e-18, -4.20128342e-17, 2.08166817e-17, 2.94902991e-17, 6.24500451e-17])
        cTH = np.array([2.12082170e-05, 1.29405965e-06, -6.13606755e-06, 4.29222261e-05, -1.04934521e-04, 8.65209674e-05])
        cTP = np.array([7.85881100e-07, -1.97232615e-08, 3.87305157e-08, -6.04645236e-08, 6.25595229e-08, -2.94970993e-08])
        cHP = np.array([-1.40967131e-16, 1.64663205e-18, -7.48099499e-18, 8.67361738e-18, -6.93889390e-18, -1.73472348e-18])
    elif wvrange == 1:
        cref = np.array([0.200192e-3, 0.113474e-9, -0.424595e-14, 0.100957e-16, -0.293315e-20, 0.307228e-24])  # cm^i
        cT = np.array([0.588625e-1, -0.385766e-7, 0.888019e-10, -0.567650e-13, 0.166615e-16, -0.174845e-20])  # K cm^i
        cTT = np.array([-3.01513, 0.406167e-3, -0.514544e-6, 0.343161e-9, -0.101189e-12, 0.106749e-16])  # K^2 cm^i
        cH = np.array([-0.103945e-7, 0.136858e-11, -0.171039e-14, 0.112908e-17, -0.329925e-21, 0.344747e-25])  # cm^i / %
        cHH = np.array([0.573256e-12, 0.186367e-16, -0.228150e-19, 0.150947e-22, -0.441214e-26, 0.461209e-30])  # cm^i / %^2
        cP = np.array([0.267085e-8, 0.135941e-14, 0.135295e-18, 0.818218e-23, -0.222957e-26, 0.249964e-30])  # cm^i / Pa
        cPP = np.array([0.609186e-17, 0.519024e-23, -0.419477e-27, 0.434120e-30, -0.122445e-33, 0.134816e-37])  # cm^i / Pa^2
        cTH = np.array([0.497859e-4, -0.661752e-8, 0.832034e-11, -0.551793e-14, 0.161899e-17, -0.169901e-21])  # cm^i K / %
        cTP = np.array([0.779176e-6, 0.396499e-12, 0.395114e-16, 0.233587e-20, -0.636441e-24, 0.716868e-28])  # cm^i K / Pa
        cHP = np.array([-0.206567e-15, 0.106141e-20, -0.149982e-23, 0.984046e-27, -0.288266e-30, 0.299105e-34])  # cm^i / Pa %
    elif wvrange == 2:
        cref = np.array([0.200049e-3, 0.145221e-9, 0.250951e-12, -0.745834e-15, -0.161432e-17, 0.352780e-20])  # cm^i
        cT = np.array([0.588431e-1, -0.825182e-7, 0.137982e-9, 0.352420e-13, -0.730651e-15, -0.167911e-18])  # K cm^i
        cTT = np.array([-3.13579, 0.694124e-3, -0.500604e-6, -0.116668e-8, 0.209644e-11, 0.591037e-14])  # K^2 cm^i
        cH = np.array([-0.108142e-7, 0.230102e-11, -0.154652e-14, -0.323014e-17, 0.630616e-20, 0.173880e-22])  # cm^i / %
        cHH = np.array([0.586812e-12, 0.312198e-16, -0.197792e-19, -0.461945e-22, 0.788398e-25, 0.245580e-27])  # cm^i / %^2
        cP = np.array([0.266900e-8, 0.168162e-14, 0.353075e-17, -0.963455e-20, -0.223079e-22, 0.453166e-25])  # cm^i / Pa
        cPP = np.array([0.608860e-17, 0.461560e-22, 0.184282e-24, -0.524471e-27, -0.121299e-29, 0.246512e-32])  # cm^i / Pa^2
        cTH = np.array([0.517962e-4, -0.112149e-7, 0.776507e-11, 0.172569e-13, -0.320582e-16, -0.899435e-19])  # cm^i K / %
        cTP = np.array([0.778638e-6, 0.446396e-12, 0.784600e-15, -0.195151e-17, -0.542083e-20, 0.103530e-22])  # cm^i K / Pa
        cHP = np.array([-0.217243e-15, 0.104747e-20, -0.523689e-23, 0.817386e-26, 0.309913e-28, -0.363491e-31])  # cm^i / Pa %

    # use numpy arrays to calculate all the coefficients at the same time
    coeff = cref[i] + cT[i] * dT + cTT[i] * (dT ** 2) + cH[i] * dH + cHH[i] * (dH ** 2) + cP[i] * dP + cPP[i] * (
                dP ** 2) + cTH[i] * dT * dH + cTP[i] * dT * dP + cHP[i] * dH * dP

    return coeff

def nMatharmini(wv, P, T, H):
    """
    Calculate the index of refraction from 0.7 µm to 1.36 µm as given by Noel et al. (2023). 
    

    Inputs:
        wv: wavelength in microns
        P:  Pressure in Pa
        T:  Temperature in Kelvin
        H:  relative humidity in % (i.e. between 0 and 100)
    Return:
        n:  index of refraction
    """
    # Defining the reference values
    referencewavelen = 0.77 # Microns
    reftemp = 280.65 # Kelvin
    pressureref = 66625 # Pa
    refhumidity = 50 # %

    # The Taylor expansion coefficients
    coefs = np.array([0.000185566259, 0.0533344343, 4.37191645e-06, -5.29847992e-09, 1.7263833e-13, 2.7897497e-09,
                      2.26729683e-17, 2.1208217e-05, 7.858811e-07, -1.40967131e-16, -4.68511206e-06, -0.00124712782,
                      -6.25121335e-08, -3.13820651e-10, 1.61933914e-12, -7.00536198e-11, 7.56136386e-18, 1.29405965e-06,
                      -1.97232615e-08, 1.64663205e-18, 9.19681919e-06, 0.00233119745, 1.63938942e-07, 4.69827651e-10,
                      -5.64003179e-12, 1.37565581e-10, -4.20128342e-17, -6.13606755e-06, 3.87305157e-08, -7.48099499e-18,
                      -1.44638085e-05, -0.00232913516, -2.11103761e-07, -3.50677283e-09, -2.62670875e-12, -2.14757969e-10,
                      2.08166817e-17, 4.29222261e-05, -6.04645236e-08, 8.67361738e-18, 1.52286899e-05, 1.75945139e-06,
                      -1.52898469e-08, 9.63769669e-09, 1.211447e-11, 2.22197137e-10, 2.94902991e-17, -0.000104934521,
                      6.25595229e-08, -6.9388939e-18, -7.42131053e-06, 0.00151989359, 1.13124404e-07, -9.13487764e-09,
                      4.26582641e-12, -1.04766954e-10, 6.24500451e-17, 8.65209674e-05, -2.94970993e-08, -1.73472348e-18])

    # Using the equation in the paper
    variables = []
    for power in range(0, 6):
        wvcoef = (wv - referencewavelen) ** power

        variables.append(wvcoef)
        variables.append(((1 / T) - (1 / reftemp)) * wvcoef)
        variables.append((((1 / T) - (1 / reftemp)) ** 2) * wvcoef)
        variables.append((H - refhumidity) * wvcoef)
        variables.append(((H - refhumidity) ** 2) * wvcoef)
        variables.append((P - pressureref) * wvcoef)
        variables.append(((P - pressureref) ** 2) * wvcoef)
        variables.append((((1 / T) - (1 / reftemp)) * (H - refhumidity)) * wvcoef)
        variables.append((((1 / T) - (1 / reftemp)) * (P - pressureref)) * wvcoef)
        variables.append(((H - refhumidity) * (P - pressureref)) * wvcoef)

    variables = np.array(variables)
    # Multiplying everything together and summing
    return 1+np.sum(coefs*variables)
