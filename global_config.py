# Missing ratio for masked values
missing_ratio = 0.3
## Default value to replace the nan before being feeded to NN
val_spec = 0
## Random seed
random_state = 42
# List of statistical methods covered
class_methods = [
    "mean",
    "median",
    "mode",
    "LOCF",
    "NOCB",
    "linear_interpolation",
    "spline_interpolation",
    "knn",
    "mice",
]
