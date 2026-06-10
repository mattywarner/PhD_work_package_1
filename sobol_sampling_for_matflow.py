def _values_from_sobol(
        cls,
        paths: Sequence[str],
        dim_num_samples: int,
        *,
        bounds: dict[str, dict[str, str | Sequence[float]]] | None = None,
        scramble: bool = True,
        bits: int = None,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        rng=None
) -> NDArray:
    
    from scipy.stats.qmc import Sobol, scale

    num_paths = len(paths)
    kwargs = dict(
        d = num_paths,
        scramble = scramble,
        bits = bits,
        optimization = optimization,
        rng = rng
    )

    bounds = bounds or {}

    scaling = np.asarray(
        [bounds.get(path, {}).get("scaling","linear") for path in paths]
    )

    all_extents = [bounds.get(path, {}).get("extent", [0,1]) for path in paths]

    extent = np.asarray(
        [
        np.log10(all_extents[i] if scaling[i] == "log" else all_extents[i] for i in range(len(scaling)))
        ]
    ).T

    try:
        sampler = Sobol(**kwargs)

    except:
        pass

    sobol_samples = scale(
        sampler.random_base2(m = dim_num_samples), l_bounds=extent[0], u_bounds = extent[1]
        )
    
    for i in range(len(scaling)):
        if scaling[i] == "log":
            sobol_samples[:,i] = 10**sobol_samples[:,i]

    return sobol_samples.T

@classmethod
def from_sobol(
    cls,
    paths: Sequence[str],
    dim_num_samples: int,
    *,
    bounds: dict[str, dict[str, str | Sequence[float]]] | None = None,
    scramble: bool = True,
    bits: int = None,
    optimization: Literal["random-cd", "lloyd"] | None = None,
    rng=None,
    nesting_order: int | float | None = None,
    label: str | int | None = None,
) -> Self:
    
    kwargs = {
        "paths": paths,
        "dim_num_samples": dim_num_samples,
        "scramble": scramble,
        "bits": bits,
        "optimization": optimization,
        "rng": rng,
        "bounds": bounds,
    }

    values = cls._values_from_sobol(**kwargs)
    assert values is not None
    obj = cls(
        paths=paths,
        values=values,
        nesting_order=nesting_order,
        label=label
    )
    obj._values_method = "from_sobol"
    obj._values_method_args = kwargs
    
    return obj