import uproot
import hist
import numpy as np


def save_histograms(histogram, filename, add_offset=False):
    with uproot.recreate(filename) as f:
        # save all available histograms to disk
        #for channel, histogram in hist_dict.items():
        # optionally add minimal offset to avoid completely empty bins
        # (useful for the ML validation variables that would need binning adjustment
        # to avoid those)
        if add_offset:
            histogram += 1e-6
            # reference count for empty histogram with floating point math tolerance
            empty_hist_yield = histogram.axes[0].size*(1e-6)*1.01
        else:
            empty_hist_yield = 0

        for sample in histogram.axes[1]:
            for variation in histogram[:, sample, :].axes[1]:
                # if pt_scale_up or pt_res_up, add another histogram which is nominal - (variation - nominal) as down variation
                if variation in ["pt_scale_up", "pt_res_up", "ME_var", "PS_var"]:
                    #new_variation = variation.replace("_up", "Down")
                    #new_values = histogram[:, sample, "nominal"].values() - (
                    #    histogram[:, sample, variation].values()
                    #    - histogram[:, sample, "nominal"].values()
                    #)
                    #new_variances = histogram[:, sample, "nominal"].variances() + (
                    #    histogram[:, sample, variation].variances()
                    #    - histogram[:, sample, "nominal"].variances()
                    #)
                    #new_hist = hist.Hist(*histogram.axes[:1], storage=hist.storage.Weight())
                    #new_hist[...] = np.array(list(zip(new_values, new_variances)), dtype=new_hist.view().dtype)
                    #f[f"{sample}_{new_variation}"] = new_hist
                    diff_with_nominal = (histogram[:, sample, variation].values() - histogram[:, sample, "nominal"].values()) / 2
                    diff_with_nominal_variances = (histogram[:, sample, variation].variances() + histogram[:, sample, "nominal"].variances()) / 4
                    hist_up = histogram[:, sample, "nominal"].values() + diff_with_nominal
                    hist_down = histogram[:, sample, "nominal"].values() - diff_with_nominal
                    hist_up_variances = histogram[:, sample, "nominal"].variances() + diff_with_nominal_variances
                    hist_down_variances = histogram[:, sample, "nominal"].variances() + diff_with_nominal_variances
                    new_variation_up = variation.replace("_up", "Up") if variation.endswith("_up") else variation + "Up"
                    new_variation_down = variation.replace("_up", "Down") if variation.endswith("_up") else variation + "Down"
                    new_hist_up = hist.Hist(*histogram.axes[:1], storage=hist.storage.Weight())
                    new_hist_up[...] = np.array(list(zip(hist_up, hist_up_variances)), dtype=new_hist_up.view().dtype)
                    new_hist_down = hist.Hist(*histogram.axes[:1], storage=hist.storage.Weight())
                    new_hist_down[...] = np.array(list(zip(hist_down, hist_down_variances)), dtype=new_hist_down.view().dtype)
                    f[f"{sample}_{new_variation_up}"] = new_hist_up
                    f[f"{sample}_{new_variation_down}"] = new_hist_down 
                else:
                    pass
                
            for variation in histogram[:, sample, :].axes[1]:
                variation_string = "" if variation == "nominal" else f"_{variation}"
                current_1d_hist = histogram[:, sample, variation]

                if sum(current_1d_hist.values()) > empty_hist_yield:
                    # only save histograms containing events
                    # many combinations are not used (e.g. ME var for W+jets)
                    
                    # in variation_string, replace '_up' with 'Up' and '_down' with 'Down' for stupid combine naming
                    variation_string = variation_string.replace("_up", "Up").replace("_down", "Down").replace("scaleup", "scaleUp").replace("scaledown", "scaleDown")
                    f[f"{sample}{variation_string}"] = current_1d_hist


        # add pseudodata histogram if all inputs to it are available
        if (
            sum(histogram[:, "ttbar", "ME_var"].values()) > empty_hist_yield
            and sum(histogram[:, "ttbar", "PS_var"].values()) > empty_hist_yield
            and sum(histogram[:, "wjets", "nominal"].values()) > empty_hist_yield
        ):
            f["data_obs"] = (
                histogram[:, "ttbar", "ME_var"] + histogram[:, "ttbar", "PS_var"]
            ) / 2 + histogram[:, "wjets", "nominal"]
