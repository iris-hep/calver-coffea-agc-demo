# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.9
# ---

# %% [markdown]
# # Statistical Inference with Combine

# %%
from IPython.display import display, IFrame
import os

# %%
os.makedirs("combine_plots", exist_ok=True)

# %%
# !cat datacard_by_hand.txt

# %%
# !text2workspace.py datacard_by_hand.txt --PO 'map=.*/ttbar:r[1.0,0.0,3.0]'

# %% [markdown]
# ## Impacts

# %%
# needed because of https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/issues/1049
os.environ["CMSSW_BASE"] = "."
os.environ["SCRAM_ARCH"] = "."

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --doInitialFit -m 125 --nllbackend legacy

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --doFits -m 125

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --output impacts.json -m 125

# %%
# !plotImpacts.py -i impacts.json -o combine_plots/impacts

# %%
pdf_path = "combine_plots/impacts.pdf"
display(IFrame(pdf_path, width=800, height=600))

# %% [markdown]
# ## Pre- and post-fit distributions

# %%
# !combine -M FitDiagnostics datacard_by_hand.root --saveShapes --saveWithUncertainties -n FitDiagnosticsStuff

# %%
# !for region in bin4j1b bin4j2b; do for shape in shapes_prefit shapes_fit_b shapes_fit_s; do python3 combine_scripts/postFitPlot_new.py --input_file fitDiagnosticsFitDiagnosticsStuff.root --shape_type $shape --region $region; done; done

# %%
display(IFrame("combine_plots/stacked_plot_shapes_prefit_bin4j1b.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_fit_s_bin4j1b.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_prefit_bin4j2b.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_fit_s_bin4j2b.png", width=800, height=600))

# %% [markdown]
# ## Likelihood scan for mu

# %%
# !combine -M MultiDimFit datacard_by_hand.root -n .datacard_by_hand.snapshot --rMin -1 --rMax 4 --saveWorkspace

# %%
# !combine -M MultiDimFit higgsCombine.datacard_by_hand.snapshot.MultiDimFit.mH120.root -n .datacard_by_hand --rMin 0 --rMax 2 --algo grid --points 80 --snapshotName MultiDimFit

# %%
# !combine -M MultiDimFit higgsCombine.datacard_by_hand.snapshot.MultiDimFit.mH120.root -n .datacard_by_hand.freezeAll --rMin 0 --rMax 2 --algo grid --points 800 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances

# %%
# !python3 combine_scripts/plot1DScan.py higgsCombine.datacard_by_hand.MultiDimFit.mH120.root --others 'higgsCombine.datacard_by_hand.freezeAll.MultiDimFit.mH120.root:FreezeAll:2' -o combine_plots/likelihood_scan --breakdown Syst,Stat

# %%
display(IFrame("combine_plots/likelihood_scan.png", width=800, height=600))
