import numpy as np
import matplotlib.pyplot as plt

# Given parameters
W = 2400  # Aircraft weight in lbs
CD0 = 0.0317  # Zero-lift drag coefficient
AR = 5.71  # Aspect Ratio
e0 = 0.6  # Oswald efficiency factor
CL_max = 1.45  # Maximum lift coefficient without flaps
BHP_SL = 180  # Brake Horsepower at sea level
eta_p = 0.8  # Assumed propeller efficiency
S = 160  # Estimated wing area in sq ft 
k = 1 / (np.pi * e0 * AR)
velocities = np.arange(70,271,1)
density_table = {
    0: 0.002377,
    5000: 0.002048,
    10000: 0.001755,
    15000: 0.001497,
}

# Altitudes to evaluate (in feet)
altitudes = [0, 5000, 10000, 15000]  # in ft
all_altitudes = np.arange(0,17001,200) 

# Standard Atmosphere Properties at altitudes
rho_0 = 0.002377  # Sea level air density in slug/ft^3
g = 32.174  # Gravitational acceleration ft/s^2



def calculate_rate_of_climb(v, alt, rho):
    """
    Calculate the rate of climb for a given velocity and altitude.
    """
    D = get_drag(v, rho)
    P_av = BHP_SL * get_density_ratio(alt) * eta_p * 550
    P_req = D * v
    RC = (P_av - P_req) / W
    # Convert RC from ft/s to ft/min 
    RC_fpm = RC * 60
    return RC_fpm

def convert_ft_per_s_to_knots(velocities_ft_s):
    """Convert velocities from ft/s to knots"""
    return [v * 0.5925 for v in velocities_ft_s]


def get_drag(v, rho):
    return 0.5*rho*(v**2)*S*get_cd(v,rho)


def get_density_ratio(altitude):
    """Returns the density ratio at a given altitude based on the standard atmosphere."""
    rho = density_table.get(altitude, calculate_air_density(altitude))
    return rho / rho_0

def get_cl(v, rho):
    return 2*W / (rho * v**2 * S)

def get_cd(v, rho):
    return CD0 + k * (get_cl(v,rho))**2

def get_cl_max_l_d():
    return np.sqrt(CD0/k)



def convert_v_to_veq(velocities, alt):
    return [v * np.sqrt(get_density_ratio(alt)) for v in velocities]

def calculate_air_density(altitude_ft):
    altitude_m = altitude_ft * 0.3048
    T0 = 288.15
    P0 = 101325
    g0 = 9.80665
    R = 287.058
    L = 0.0065
    T = T0 - (L * altitude_m)
    exponent = (g0 / (R * L))
    P = P0 * (1 - (L * altitude_m) / T0) ** exponent
    rho_metric = P / (R * T)
    rho_imperial = rho_metric * 0.00194032
    return rho_imperial

def get_performance_data_part_a(alt):
    RC_data = {}
    Veq_data = {}
    for a in alt:
        if a in density_table:
            rho = density_table[a]
        else:
            rho = calculate_air_density(a)
        
        RC_values = []
        
        for v_true in velocities:
            RC_fpm = calculate_rate_of_climb(v_true, a, rho)
            RC_values.append(RC_fpm)
        
        veq = convert_v_to_veq(velocities, a)
        veq_knots = convert_ft_per_s_to_knots(veq)
        
        RC_data[a] = RC_values
        Veq_data[a] = veq_knots
    return RC_data, Veq_data

def get_performance_data_part_c(alt):
    RC_max_data = {}
    for a in alt:
        rho = calculate_air_density(a)
        
        RC_values = []
        for v in velocities:
            RC_fpm = calculate_rate_of_climb(v, a, rho)
            RC_values.append(RC_fpm)
        
        RC_max_data[a] = max(RC_values)

    return dict(sorted(RC_max_data.items()))

def main():
    # Calculate all data once

    # for part a and b
    RC_data, Veq_data = get_performance_data_part_a(altitudes)
    RC_max_data = get_performance_data_part_c(all_altitudes)
   

   
    plt.figure(figsize=(10,6))
    altitudes_list = list(RC_max_data.keys())
    RC_max_list = list(RC_max_data.values())
    plt.plot(altitudes_list, RC_max_list, label='Rate of Climb')
    plt.xlabel('Altitude (ft)')
    plt.ylabel('Rate of Climb (ft/min)')
    plt.title('Rate of Climb vs Altitude')
    plt.ylim(top=1500, bottom=0)  
    plt.grid(True)
    plt.legend()
    plt.show()

    # for part c

    
    # Part a: RC vs Velocity
    plt.figure(figsize=(10, 6))
    for alt in altitudes:
        plt.plot(Veq_data[alt], RC_data[alt], label=f'Altitude: {alt} ft')
    
    plt.xlabel('Horizontal Velocity (knots)')
    plt.ylabel('Rate of Climb (ft/min)')
    plt.title('Rate of Climb vs Velocity at Different Altitudes')
    plt.ylim(bottom=-100)  
    plt.grid(True)
    plt.legend()
    plt.show()

    # Part b: Velocity vs RC
    plt.figure(figsize=(10, 6))
    for alt in altitudes:
        plt.plot(RC_data[alt], Veq_data[alt], label=f'Altitude: {alt} ft')
    
    plt.xlabel('Rate of Climb (ft/min)')
    plt.ylabel('Horizontal Velocity (knots)')
    plt.title('Velocity vs Rate of Climb at Different Altitudes')
    plt.xlim(left=-100)  
    plt.grid(True)
    plt.legend()
    plt.show()
    



if __name__ == "__main__":
    main()