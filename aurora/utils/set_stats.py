from aurora.normalisation import locations, scales

# For any feature name you plan on using, set their mean and stds here.

## BIG DISCLAIMER: "The issue is that the static variable names are concatenated to the surface variable names, and the position in the resulting list is used to identify the parameters. Consequently, adding in one more surface variable shifts the positions of the static variables:"
# https://github.com/microsoft/aurora/blob/c81800565cfd72b963ce8834de5c2389e9f59859/aurora/model/encoder.py#L80

# Normalisation means:
locations["new_surf_var"] = 0.0
locations["new_static_var"] = 0.0
locations["new_atmos_var"] = 0.0

# Normalisation standard deviations:
scales["new_surf_var"] = 1.0
scales["new_static_var"] = 1.0
scales["new_atmos_var"] = 1.0

