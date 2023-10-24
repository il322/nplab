'''
Renishaw laser power dictionary for easy import and access
'''

all_laser_powers = {
                    532:#measured 2022-08-10 w/ 5x objective
                    {100.     :  46.0,
                     50.      :  23.0,
                     10.      :   4.80,
                      5.      :   2.25,
                      1.      :   0.330,
                      0.5     :   0.159,
                      0.1     :   0.032,
                      0.05    :   0.015,
                      0.0001  :   0.032/100,#extrapolated
                      0.00005 :   0.015/100,#extrapolated
                      0.00001 :   0.032/1000#extrapolated
                      },
                    633:#measured 2022-08-10 w/ 5x objective
                    {100.     :  3.47,
                     50.      :  1.60,
                     10.      :  0.325,
                      5.      :  0.155,
                      1.      :  0.0300,
                      0.5     :  0.0140,
                      0.1     :  0.00287,
                      0.05    :  0.00140,
                      0.0001  :  0.00287/100,#extrapolated
                      0.00005 :  0.00140/100,#extrapolated
                      0.00001 :  0.00287/1000#extrapolated
                      },
                    785:#measured 2023-03-23 w/20x objective
                    {100.      : 131.,
                     50.      :  61.1,
                     10.      :  22.8,
                      5.      :  10.6,
                      1.      :  4.54,
                      0.5     :  2.14,
                      0.1     :  0.812,
                      0.05    :  0.382,
                      0.0001  :  0.00725,
                      0.00005 :  0.00340,
                      0.00001 :  0.00129
                      },
                    }