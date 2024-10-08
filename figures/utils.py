"""
colors = {
    "fene": "blue",
    "stacking": "red",
    "hydrogen_bonding": "purple",
    "cross_stacking": "orange",
    "coaxial_stacking": "green",
    "all": "black"
}
"""
colors = {
    "fene": "#56B4E9",
    "stacking": "#009E73",
    "hydrogen_bonding": "#F0E442",
    "cross_stacking": "#0072B2",
    "coaxial_stacking": "#D55E00",
    "all": "#000000"
}

labels = {
    "fene": "FENE",
    "stacking": "Stacking",
    "hydrogen_bonding": "Hydrogen Bonding",
    "cross_stacking": "Cross Stacking",
    "coaxial_stacking": "Coaxial Stacking",
    "all": "All"
}



label_dict = {
    "fene": {
        "r0_backbone": r'$\delta r^0_{backbone}$',
        "delta_backbone": r'$\Delta_{backbone}$',

    },
    "cross_stacking": {
        "r0_cross": r'$\delta r^0_{cross}$',
        "dr_c_cross": r'$\delta r^c_{cross}$',
        "theta0_cross_1": r'$\theta^0_{cross,1}$',
        "theta0_cross_4": r'$\theta^0_{cross,4}$'
    },
    "stacking": {
        "dr0_stack": r'$\delta r^0_{stack}$',
        "dr_c_stack": r'$\delta r^c_{stack}$',
        "theta0_stack_5": r'$\theta^0_{stack,5}$',
        "theta0_stack_4": r'$\theta^0_{stack,4}$',
        "eps_stack_base": r'$\varepsilon_{stack}^{base}$',
        "a_stack_4": r'$a_{stack,4}$',
    },
    "hydrogen_bonding": {
        "dr0_hb": r'$\delta r^0_{hb}$',
        "theta0_hb_7": r'$\theta^0_{hb,7}$'
    }
}
