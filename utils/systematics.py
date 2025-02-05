import awkward as ak
import numpy as np

def rand_gauss(item):
    # from https://github.com/scikit-hep/coffea/blob/master/src/coffea/jetmet_tools/CorrectedJetsFactory.py#L42
    seeds = (
        ak.flatten(ak.typetracer.length_one_if_typetracer(item)).to_numpy().view("i4")
    )
    randomstate = np.random.Generator(np.random.PCG64(seeds))

    def getfunction(layout, depth, **kwargs):
        if isinstance(layout, ak.contents.NumpyArray) or not isinstance(
            layout, (ak.contents.Content,)
        ):
            return ak.contents.NumpyArray(
                randomstate.normal(loc=1, scale=0.05, size=len(layout)).astype(np.float32)
            )
        return None

    out = ak.transform(
        getfunction,
        ak.typetracer.length_zero_if_typetracer(item),
        behavior=item.behavior,
    )
    if ak.backend(item) == "typetracer":
        out = ak.Array(
            out.layout.to_typetracer(forget_length=True), behavior=out.behavior
        )

    assert out is not None
    return out